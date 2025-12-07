// Script.js
// Search Books Function



// script.js - pure front-end TF.js recommender (no Flask)

/* --------------------
   Global state & load USE
   -------------------- */
let useModel = null;
let rankingModel = null; // will be created per-query (lightweight)
const EMB_DIM = 512;     // USE dimension

async function ensureUSE() {
  if (!useModel) {
    document.getElementById('model-status').innerText = 'Loading USE (this may take a few seconds)...';
    useModel = await use.load();
    document.getElementById('model-status').innerText = 'USE loaded';
  }
  return useModel;
}

/* --------------------
   PRELOAD USE ON PAGE LOAD
   -------------------- */
window.addEventListener('load', () => {
  ensureUSE().catch(err => {
    console.error('Failed to load USE:', err);
    document.getElementById('model-status').innerText = 'Failed to load USE';
  });
});

/* --------------------
   Helper: build small ranking model
   Input features: concat(queryEmb, itemEmb, |diff|, prod) => dense => score
   -------------------- */
function buildRankingModel() {
  // Input shape: [featureDim], where featureDim = 4 * EMB_DIM
  const featureDim = EMB_DIM * 4;
  const input = tf.input({ shape: [featureDim] });

  // Small network
  let x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input);
  x = tf.layers.dropout({ rate: 0.2 }).apply(x);
  x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x);
  x = tf.layers.dense({ units: 16, activation: 'relu' }).apply(x);
  const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x);

  const m = tf.model({ inputs: input, outputs: out });
  m.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy' });
  return m;
}

/* --------------------
   Create features for each item using query & item embeddings
   features = concat(query, item, abs(query - item), query * item)
   -------------------- */
function makeFeatures(queryEmbTensor, itemEmbTensor) {
  return tf.tidy(() => {
    // queryEmbTensor: [EMB_DIM]
    // itemEmbTensor:   [N, EMB_DIM]
    const q = queryEmbTensor.reshape([1, EMB_DIM]);   // [1,512]
    const qTiled = q.tile([itemEmbTensor.shape[0], 1]); // [N,512]

    const absdiff = qTiled.sub(itemEmbTensor).abs(); // [N,512]
    const prod = qTiled.mul(itemEmbTensor);         // [N,512]

    return tf.concat([qTiled, itemEmbTensor, absdiff, prod], 1); // [N, 2048]
  });
}

/* --------------------
   Main recommendation function (runs per query)
   - Embeds query + items
   - Creates features
   - Creates rankingModel (fresh)
   - Trains briefly with weak labels (use OpenLibrary rank as weak label)
   - Predicts scores and returns ranked items
   -------------------- */
async function recommendForQuery(query, rawBooks) {
  await ensureUSE();

  // Build texts for embedding
  const itemTexts = rawBooks.map(b => `${b.title} ${b.categories || ''} ${b.author || ''}`);
  const queryText = query;

  // get embeddings
  const embeddings = await useModel.embed(itemTexts);    // [N, EMB_DIM]
  const queryEmb = await useModel.embed([queryText]);   // [1, EMB_DIM]
  const queryEmbVec = tf.squeeze(queryEmb, [0]);        // [EMB_DIM]

  // Build features tensor
  const features = makeFeatures(queryEmbVec, embeddings); // [N, 4*EMB_DIM]

  // Create weak labels:
  // Use OpenLibrary ordering: treat the first item as positive (1), others 0.
  // You can choose other heuristics (e.g., top-2 positives) depending on UX.
  const N = rawBooks.length;
  const labels = tf.tensor1d(rawBooks.map((_, i) => (i === 0 ? 1 : 0)), 'float32').reshape([N, 1]);

  // Build or recreate a fresh ranking model for this query
  if (rankingModel) {
    try { rankingModel.dispose(); } catch (e) { /*ignore*/ }
  }
  rankingModel = buildRankingModel();

  // Train briefly on the small set (fast). This lets the network learn a query-aware ranking function.
  // Note: we train for a few epochs — this is tiny (N ~ 10) so it's quick.
  // Wrap tensors in tidy to prevent leaks; but keep features/labels for training.
  await rankingModel.fit(features, labels, {
    epochs: 5,           // small number; adjust for speed/quality
    batchSize: Math.min(6, N),
    verbose: 0
  });

  // Predict scores
  const pred = rankingModel.predict(features); // [N,1]
  const scores = await pred.data();

  // Clean up tensors we created (embeddings and queryEmb are managed by USE)
  tf.dispose([embeddings, queryEmb, features, labels, pred, queryEmbVec]);

  // Merge scores with books and return ranked list
  const results = rawBooks.map((b, i) => ({ ...b, score: scores[i] }));
  results.sort((a, b) => b.score - a.score);
  return results;
}

/* --------------------
   UI helpers: render books & recommendations
   -------------------- */
function renderBooks(books) {
  $('#books-list').empty();
  if (!books || books.length === 0) {
    $('#books-list').html('<div class="col"><h5>No books found</h5></div>');
    return;
  }
  books.forEach((data) => {
    const cover = data.cover_i ? `https://covers.openlibrary.org/b/id/${data.cover_i}-L.jpg` : 'img/no-cover.png';
    const author = data.author_name ? data.author_name.join(', ') : 'Unknown Author';
    const year = data.first_publish_year || 'N/A';
    $('#books-list').append(`
      <div class="col-md-3">
        <div class="card mb-3">
          <img src="${cover}" class="card-img-top" alt="Cover">
          <div class="card-body">
            <h5 class="card-title small">${escapeHtml(data.title)}</h5>
            <h6 class="card-subtitle mb-2 text-muted small">${escapeHtml(author)} | ${escapeHtml(year)}</h6>
            <a href="#" class="btn btn-dark btn-sm see-detail" data-key="${data.key}"> See Detail </a>
          </div>
        </div>
      </div>
    `);
  });
}

function renderRecommendations(recs) {
  $('#recommendations').empty();
  if (!recs || recs.length === 0) {
    $('#recommendations').html('<div class="col"><h5>No recommendations</h5></div>');
    return;
  }
  recs.forEach((b) => {
    // we will search OpenLibrary to get the full doc for cover/author/year (similar to original)
    $.ajax({
      url: 'https://openlibrary.org/search.json',
      type: 'GET',
      data: { title: b.title, limit: 1 },
      dataType: 'json',
      success: function (res) {
        if (!res.docs || res.docs.length === 0) return;
        const doc = res.docs[0];
        const cover = doc.cover_i ? `https://covers.openlibrary.org/b/id/${doc.cover_i}-L.jpg` : 'img/no-cover.png';
        const author = doc.author_name ? doc.author_name.join(', ') : 'Unknown';
        const year = doc.first_publish_year || 'N/A';
        $('#recommendations').append(`
          <div class="col-md-3 mb-3">
            <div class="card">
              <img src="${cover}" class="card-img-top">
              <div class="card-body">
                <h5 class="card-title small">${escapeHtml(doc.title)}</h5>
                <h6 class="card-subtitle mb-2 text-muted small">${escapeHtml(author)} | ${escapeHtml(year)}</h6>
                <a href="#" class="btn btn-dark btn-sm see-detail" data-key="${doc.key}"> See Detail </a>
                <div class="mt-1"><small>score: ${Number(b.score).toFixed(3)}</small></div>
              </div>
            </div>
          </div>
        `);
      }
    });
  });
}

/* --------------------
   Search flow
   -------------------- */
async function searchBooks() {
  $('#books-list').html('');
  $('#recommendations').html('');
  const q = $('#search-input').val().trim();
  if (!q) { alert('Please enter a search query'); return; }

  $('#status').text('Searching OpenLibrary...');
  try {
    const res = await $.ajax({
      url: 'https://openlibrary.org/search.json',
      type: 'GET',
      dataType: 'json',
      data: { title: q, limit: 12 }
    });

    if (!res.docs || res.docs.length === 0) {
      $('#books-list').html('<div class="col"><h5>No books found</h5></div>');
      $('#status').text('No results');
      return;
    }

    // show raw results
    renderBooks(res.docs);

    // Build a simple cleaned book array for embedding
    const booksForModel = res.docs.map(d => ({
      title: d.title,
      author: d.author_name ? d.author_name.join(', ') : 'Unknown',
      categories: d.subject ? d.subject.join(' ') : '',
      key: d.key,
      raw: d
    }));

    $('#status').text('Computing embeddings and ranking...');
    // Recommend using TF.js deep model (query-aware)
    const ranked = await recommendForQuery(q, booksForModel);

    // Show top 4 recommended items
    renderRecommendations(ranked.slice(0, 4));
    $('#status').text('Done');

  } catch (err) {
    console.error('Search error', err);
    $('#books-list').html('<div class="col"><h5>Error fetching data</h5></div>');
    $('#status').text('Error');
  }
}

/* --------------------
   Modal detail handler (fetch work details)
   -------------------- */
$('#books-list, #recommendations').on('click', '.see-detail', function (e) {
  e.preventDefault();
  const workKey = $(this).data('key');
  if (!workKey) return;
  $('#modal-title').text('Loading...');
  $('#modal-body').html('Loading...');
  $('#exampleModal').modal('show');

  $.ajax({
    url: `https://openlibrary.org${workKey}.json`,
    type: 'GET',
    dataType: 'json',
    success: function (data) {
      const cover = (data.covers && data.covers.length > 0)
        ? `https://covers.openlibrary.org/b/id/${data.covers[0]}-L.jpg`
        : 'img/no-cover.png';

      function cleanDescription(raw) {
        if (!raw) return 'No description available';
        let desc = (typeof raw === 'string') ? raw : raw.value || '';
        desc = desc.replace(/\[.*?\]\(.*?\)/g, '')
                   .replace(/\[.*?\]/g, '')
                   .replace(/https?:\/\/\S+/g, '')
                   .replace(/\s+/g, ' ').trim();
        return desc;
      }

      const description = cleanDescription(data.description);
      const categories = (data.subjects && data.subjects.length) ? data.subjects.join(', ') : 'N/A';
      const published = data.first_publish_date || data.created?.value || 'N/A';

      $('#modal-title').text(data.title || 'Detail');
      $('#modal-body').html(`
        <div class="row">
          <div class="col-md-4"><img src="${cover}" class="img-fluid"></div>
          <div class="col-md-8">
            <p><strong>Description:</strong> ${escapeHtml(description)}</p>
            <p><strong>Categories:</strong> ${escapeHtml(categories)}</p>
            <p><strong>Published:</strong> ${escapeHtml(published)}</p>
          </div>
        </div>
      `);
    },
    error: function () {
      $('#modal-body').html('<p>Error loading details.</p>');
    }
  });
});

/* --------------------
   Utilities
   -------------------- */
function escapeHtml(unsafe) {
  return (unsafe + '').replace(/[&<"'>]/g, function (m) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m];
  });
}

/* --------------------
   Bindings
   -------------------- */
$('#search-button').on('click', searchBooks);
$('#search-input').on('keyup', function(e) { if (e.which === 13) searchBooks(); });

/* --------------------
   Initial: preload USE to improve first-run latency
   -------------------- */
ensureUSE().catch(err => {
  console.error('Failed to load USE:', err);
  document.getElementById('model-status').innerText = 'Failed to load USE';
});

// End of script.js
