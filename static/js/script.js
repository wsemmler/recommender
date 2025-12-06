// Script.js
// Search Books Function
function searchBooks() {
    $('#books-list').html('');
    $('#recommendations').html('');

    let query = $('#search-input').val().trim();
    if (!query) {
        alert('Please enter a book title');
        return;
    }

    $.ajax({
        url: 'https://openlibrary.org/search.json',
        type: 'get',
        dataType: 'json',
        data: { title: query, limit: 12 },
        success: function (result) {
            if (result.docs && result.docs.length > 0) {
                let books = result.docs;

                // CLEAN BOOKS FOR PYTHON API
                let booksForPython = books.map(data => ({
                    title: data.title,
                    key: data.key,
                    author: data.author_name ? data.author_name.join(', ') : 'Unknown Author',
                    year: data.first_publish_year || 'N/A',
                    categories: data.subject ? data.subject.join(', ') : ''
                }));
                // AJAX SEND TO PYTHON
                $('#books-results').html("");
                $.each(books, function (i, data) {
                    let cover = data.cover_i
                        ? `https://covers.openlibrary.org/b/id/${data.cover_i}-L.jpg`
                        : 'img/no-cover.png';
                    let author = data.author_name ? data.author_name.join(', ') : 'Unknown Author';
                    let year = data.first_publish_year || 'N/A';

                    $('#books-list').append(`
                        <div class="col-md-3">
                            <div class="card mb-3">
                                <img src="${cover}" class="card-img-top" alt="Cover">
                                <div class="card-body">
                                    <h5 class="card-title small">${data.title}</h5>
                                    <h6 class="card-subtitle mb-2 text-muted small">${author} | ${year}</h6>
                                    <a href="#" class="btn btn-dark btn-sm see-detail" data-toggle="modal" 
                                    data-target="#exampleModal" data-key="${data.key}"> See Detail </a>
                                </div>
                            </div>
                        </div>
                    `);
                });
                // CALL PYTHON RECOMMENDER API HERE
                sendToPythonForRecommendation(booksForPython);
                
                $('#search-input').val('');

            } else {
                $('#books-list').html(`
                    <div class="col">
                        <h1 class="text-center">No Books Found!</h1>
                    </div>
                `);
            }
        },
        error: function () {
            $('#books-list').html(`
                <div class="col">
                    <h3 class="text-center">Error fetching data</h3>
                </div>
            `);
        }
    });
}

// ===============================
// SEND BOOK LIST TO PYTHON API
// ===============================
function sendToPythonForRecommendation(books) {

    $.ajax({
        url: 'http://127.0.0.1:5000/api/recommend',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ books: books }),
        success: function (response) {
            console.log("Recommendations from Python:", response);
            displayRecommendations(response);
        },
        error: function (err) {
            console.error("Error sending to Python:", err);
        }
    });
}

// ===============================
// DISPLAY RECOMMENDATIONS
// ===============================
function displayRecommendations(data) {

    // Create section if not exists
    // if ($("#recommendations").length === 0) {
    //     $(".container").append(`
    //         <hr>
    //         <h3>Recommendations</h3>
    //         <div id="recommendations" class="row"></div>
    //     `);
    // }

    $('#recommendations-section').html(`
        <hr>
        <h3>Recommendations</h3>
        <div id="recommendations" class="row"></div>
    `);
    
    // let displayedTitles = new Set();

    data.forEach(item => {
         item.recommendations.forEach(recTitle => {
        // let displayed = false; // flag to check if we displayed a recommendation
        // for (let recTitle of item.recommendations) {
        //     if (!displayedTitles.has(recTitle)) {
        //     displayedTitles.add(recTitle);
            $.ajax({
                url: 'https://openlibrary.org/search.json',
                type: 'GET',
                data: { title: recTitle, limit: 1 },
                dataType: 'json',
                success: function(res) {
                    if (res.docs && res.docs.length > 0) {
                        let book = res.docs[0];
                        let cover = book.cover_i 
                            ? `https://covers.openlibrary.org/b/id/${book.cover_i}-L.jpg` 
                            : 'img/no-cover.png';
                        let author = book.author_name ? book.author_name.join(', ') : 'Unknown Author';
                        let year = book.first_publish_year || 'N/A';

                        $("#recommendations").append(`
                            <div class="col-md-3 mb-3">
                                <div class="card">
                                    <img src="${cover}" class="card-img-top" alt="Cover">
                                    <div class="card-body">
                                        <h5 class="card-title small">${book.title}</h5>
                                        <h6 class="card-subtitle mb-2 text-muted small">${author} | ${year}</h6>
                                        <a href="#" class="btn btn-dark btn-sm see-detail" data-toggle="modal" 
                                        data-target="#exampleModal" data-key="${book.key}"> See Detail </a>
                                    </div>
                                </div>
                            </div>
                        `);
                    }
                }
            });
            // displayed = true; 
            // break; 
        });
    });
}


// SEARCH BUTTON + ENTER KEY
$('#search-button').on('click', function () {
    searchBooks();
});
$('#search-input').on('keyup', function (e) {
    if (e.which === 13) {
        searchBooks();
    }
});


// MODAL DETAIL
$('#books-list').on('click', '.see-detail', function () {
    let workKey = $(this).data('key');

    $.ajax({
        url: `https://openlibrary.org${workKey}.json`,
        type: 'get',
        dataType: 'json',
        success: function (data) {
            // COVER
            let cover = (data.covers && data.covers.length > 0)
                ? `https://covers.openlibrary.org/b/id/${data.covers[0]}-L.jpg`
                : 'img/no-cover.png';

            // CLEAN DESCRIPTION
            function cleanDescription(rawDescription) {
                if (!rawDescription) return "No description available";

                let desc = (typeof rawDescription === "string") ? rawDescription : rawDescription.value;

                desc = desc.replace(/\[.*?\]\(.*?\)/g, '')
                        .replace(/\[.*?\]/g, '');
                desc = desc.replace(/[-]{2,}/g, ' ');
                desc = desc.replace(/https?:\/\/\S+/g, '');
                desc = desc.replace(/Contains:.*$/i, '');
                desc = desc.replace(/\s+/g, ' ').trim();

                return desc;
            }

            // CLEAN CATEGORY
            function cleanCategories(rawCategories) {
                if (!rawCategories || !rawCategories.length) return "N/A";

                // Remove numbers, codes, URLs, and "--"
                let cleaned = rawCategories
                    .map(cat => cat.replace(/--/g, ' '))            // replace '--' with space
                    .map(cat => cat.replace(/https?:\/\/\S+/g, '')) // remove URLs
                    .filter(cat => !/^[0-9\s\.\-]+$/.test(cat))     // remove pure numbers/codes
                    .map(cat => cat.trim())                          // trim whitespace
                    .filter(cat => cat.length > 0);                 // remove empty

                return cleaned.join(', ') || "N/A";
            }

            let description = cleanDescription(data.description);
            let categories = cleanCategories(data.subjects);
            let firstPublished = data.first_publish_date || "N/A";

            // PUT INTO MODAL
            $('#exampleModal .modal-title').text(data.title);
            $('#exampleModal .modal-body').html(`
                <div class="row">
                    <div class="col-md-4">
                        <img src="${cover}" class="img-fluid mb-3">
                    </div>
                    <div class="col-md-8">
                        <p><strong>Description:</strong> ${description}</p>
                        <p><strong>Category:</strong> ${categories}</p>
                        <p><strong>First Published:</strong> ${firstPublished}</p>
                    </div>
                </div>
            `);
        },
        error: function () {
            $('#exampleModal .modal-body').html('<p>Error loading details.</p>');
        }
    });
});

// End of script.js
