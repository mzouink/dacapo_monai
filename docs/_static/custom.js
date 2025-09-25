/* Custom JavaScript for DaCapo-MONAI documentation */

document.addEventListener('DOMContentLoaded', function() {
    // Add copy functionality to code blocks
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach(function(block) {
        if (!block.querySelector('.copybutton')) {
            const button = document.createElement('button');
            button.className = 'copybutton';
            button.textContent = 'Copy';
            button.addEventListener('click', function() {
                const code = block.querySelector('code') || block;
                navigator.clipboard.writeText(code.textContent).then(function() {
                    button.textContent = 'Copied!';
                    setTimeout(function() {
                        button.textContent = 'Copy';
                    }, 2000);
                });
            });
            block.style.position = 'relative';
            block.appendChild(button);
        }
    });

    // Add smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add table of contents highlighting
    const tocLinks = document.querySelectorAll('.toctree-wrapper a');
    const sections = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    
    function highlightTocLink() {
        let current = '';
        sections.forEach(function(section) {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 100) {
                current = section.id;
            }
        });
        
        tocLinks.forEach(function(link) {
            link.classList.remove('current');
            if (link.getAttribute('href').includes(current)) {
                link.classList.add('current');
            }
        });
    }
    
    window.addEventListener('scroll', highlightTocLink);
    highlightTocLink(); // Initial call

    // Add external link indicators
    const externalLinks = document.querySelectorAll('a[href^="http"]:not([href*="' + window.location.hostname + '"])');
    externalLinks.forEach(function(link) {
        link.classList.add('external-link');
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
        
        // Add external link icon
        const icon = document.createElement('span');
        icon.innerHTML = ' ↗';
        icon.className = 'external-icon';
        link.appendChild(icon);
    });

    // Add collapsible sections
    const collapsibleHeaders = document.querySelectorAll('h2, h3');
    collapsibleHeaders.forEach(function(header) {
        if (header.nextElementSibling && header.nextElementSibling.tagName !== 'H2' && header.nextElementSibling.tagName !== 'H3') {
            header.style.cursor = 'pointer';
            header.addEventListener('click', function() {
                let sibling = this.nextElementSibling;
                while (sibling && !sibling.tagName.match(/^H[2-6]$/)) {
                    sibling.style.display = sibling.style.display === 'none' ? 'block' : 'none';
                    sibling = sibling.nextElementSibling;
                }
            });
        }
    });
    
    console.log('DaCapo-MONAI documentation enhanced! 🚀');
});