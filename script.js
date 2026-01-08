// Script.js
// Search Books Function
// script.js - pure front-end TF.js recommender

let useModel = null;
let rankingModel = null; 
const EMB_DIM = 512;     

// Monitoring Elements
function createMonitoringUI() {
  if (!$('#monitoring-panel').length) {
    $('body').prepend(`
      <div id="monitoring-panel" style="position:fixed;top:0;right:0;background:#f8f9fa;border:1px solid #ddd;padding:10px;z-index:9999;font-size:12px;width:250px;">
        <div><strong>Model Status</strong></div>
        <div id="model-status">Not loaded</div>
        <div id="ranking-status">Ranking model: N/A</div>
        <div id="training-status">Training: N/A</div>
        <div id="memory-status">Memory: N/A</div>
      </div>
    `);
  }
}
createMonitoringUI();

async function ensureUSE() {
  if (!useModel) {
    document.getElementById('model-status').innerText = 'Loading USE (this may take a few seconds)...';
    useModel = await use.load();
    document.getElementById('model-status').innerText = 'USE loaded';
  }
  return useModel;
}

// PRELOAD ON PAGE LOAD

// window.addEventListener('load', () => {
//   ensureUSE().catch(err => {
//     console.error('Failed to load USE:', err);
//     document.getElementById('model-status').innerText = 'Failed to load USE';
//   });
// });

// build ranking 
function buildRankingModel() {
  const featureDim = EMB_DIM * 4;
  const input = tf.input({ shape: [featureDim] });

  let x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input);
  x = tf.layers.dropout({ rate: 0.2 }).apply(x);
  x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x);
  x = tf.layers.dense({ units: 16, activation: 'relu' }).apply(x);
  const out = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x);

  const m = tf.model({ inputs: input, outputs: out });
  m.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy' });
  $('#ranking-status').text(`Ranking model built. Layers: ${m.layers.length}`);
  return m;
}

// Create features 
function makeFeatures(queryEmbTensor, itemEmbTensor) {
  return tf.tidy(() => {
    const q = queryEmbTensor.reshape([1, EMB_DIM]);   // [1,512]
    const qTiled = q.tile([itemEmbTensor.shape[0], 1]); // [N,512]
    const absdiff = qTiled.sub(itemEmbTensor).abs(); // [N,512]
    const prod = qTiled.mul(itemEmbTensor);         // [N,512]
    return tf.concat([qTiled, itemEmbTensor, absdiff, prod], 1); // [N, 2048]
  });
}

// Memory Monitor
function updateMemoryStatus() {
  const mem = tf.memory();
  $('#memory-status').text(`Tensors: ${mem.numTensors}, Bytes: ${mem.numBytes}`);
}
setInterval(updateMemoryStatus, 2000);

// Recommend Function
async function recommendForQuery(query, rawBooks) {
  await ensureUSE();

  const itemTexts = rawBooks.map(b => `${b.title} ${b.categories || ''} ${b.author || ''}`);
  const queryText = query;

  const embeddings = await useModel.embed(itemTexts);    // [N, EMB_DIM]
  const queryEmb = await useModel.embed([queryText]);   // [1, EMB_DIM]
  const queryEmbVec = tf.squeeze(queryEmb, [0]);        // [EMB_DIM]

  const features = makeFeatures(queryEmbVec, embeddings); // [N, 4*EMB_DIM]
  const N = rawBooks.length;
  const labels = tf.tensor1d(rawBooks.map((_, i) => (i === 0 ? 1 : 0)), 'float32').reshape([N, 1]);

  if (rankingModel) {
    rankingModel.dispose();
    rankingModel = null;
    $('#ranking-status').text('Old ranking model disposed.');
  }
  rankingModel = buildRankingModel();

  await rankingModel.fit(features, labels, {
    epochs: 5,           // small number; adjust for speed/quality
    batchSize: Math.min(6, N),
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        $('#training-status').text(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}`);
      },
      onTrainEnd: () => {
        $('#training-status').text('Training done');
      }
    }
  });

  const pred = rankingModel.predict(features); // [N,1]
  const scores = await pred.data();
  console.log('Predicted scores:', scores);
  tf.dispose([embeddings, queryEmb, features, labels, pred, queryEmbVec]);

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

    renderBooks(res.docs);

    const booksForModel = res.docs.map(d => ({
      title: d.title,
      author: d.author_name ? d.author_name.join(', ') : 'Unknown',
      categories: d.subject ? d.subject.join(' ') : '',
      key: d.key,
      raw: d
    }));

    $('#status').text('Computing embeddings and ranking...');
    const ranked = await recommendForQuery(q, booksForModel);

    renderRecommendations(ranked.slice(0, 8));
    $('#status').text('Done');

  } catch (err) {
    console.error('Search error', err);
    $('#books-list').html('<div class="col"><h5>Error fetching data</h5></div>');
    $('#status').text('Error');
  }
}

// Modal detail handler (fetch work details)
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

// Utilities
function escapeHtml(unsafe) {
  return (unsafe + '').replace(/[&<"'>]/g, function (m) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m];
  });
}

// Bindings
$('#search-button').on('click', searchBooks);
$('#search-input').on('keyup', function(e) { if (e.which === 13) searchBooks(); });

// Initial: preload USE to improve first-run latency
$('button[data-toggle="pill"]').on('shown.bs.tab', async function (e) {
  const targetId = $(e.target).data('target');
  if (targetId === '#books-content' && !useLoaded) {
    try {
      document.getElementById('model-status').innerText = 'Loading USE...';
      await ensureUSE();
      useLoaded = true;
      document.getElementById('model-status').innerText = 'USE loaded';
    } catch (err) {
      console.error('Failed to load USE:', err);
      document.getElementById('model-status').innerText = 'Failed to load USE';
    }
  }
});

// End of script.js
