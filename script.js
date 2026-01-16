
document.addEventListener('DOMContentLoaded', () => {
    const grid = document.getElementById('papers-grid');
    const searchInput = document.getElementById('search-input');
    const totalCount = document.getElementById('total-count');
    const connectionCount = document.getElementById('connection-count');

    let papers = [];

    // Use local data from data.js
    papers = typeof papersData !== 'undefined' ? papersData : [];
    renderPapers(papers);

    /* 
    // Old fetch method removed for local compatibility
    async function loadPapers() { ... }
    */

    function renderPapers(filteredPapers) {
        grid.innerHTML = '';
        filteredPapers.forEach((paper, index) => {
            const card = document.createElement('div');
            card.className = 'card';
            card.style.animationDelay = `${index * 0.05}s`;

            card.innerHTML = `
                <div class="card-header">
                    <h2 class="card-title">${paper.title || 'Untitled Research'}</h2>
                    ${paper.link ? `<a href="${paper.link}" target="_blank" class="card-link">View Source ↗</a>` : ''}
                </div>
                
                <div class="card-section">
                    <div class="section-label">Main Goal</div>
                    <div class="section-content">${paper.goal || 'N/A'}</div>
                </div>

                <div class="card-section">
                    <div class="section-label">Method / Framework</div>
                    <div class="section-content">${paper.method || 'N/A'}</div>
                </div>

                <div class="card-section">
                    <div class="section-label">Key Results</div>
                    <div class="section-content">${paper.results || 'N/A'}</div>
                </div>

                <div class="connection-box">
                    <div class="section-label">Connection to Project</div>
                    <div class="section-content">${paper.connection || 'N/A'}</div>
                </div>
            `;
            grid.appendChild(card);
        });

        // Update stats
        totalCount.textContent = filteredPapers.length;
        connectionCount.textContent = filteredPapers.filter(p => p.connection && p.connection !== 'N/A').length;
    }

    searchInput.addEventListener('input', (e) => {
        const term = e.target.value.toLowerCase();
        const filtered = papers.filter(paper =>
            (paper.title && paper.title.toLowerCase().includes(term)) ||
            (paper.goal && paper.goal.toLowerCase().includes(term)) ||
            (paper.connection && paper.connection.toLowerCase().includes(term)) ||
            (paper.method && paper.method.toLowerCase().includes(term))
        );
        renderPapers(filtered);
    });

    // loadPapers(); removed
});
