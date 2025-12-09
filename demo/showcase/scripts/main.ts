import { ShowcaseScene } from './ShowcaseScene';

document.addEventListener('DOMContentLoaded', async () => {
    const showcase = new ShowcaseScene('canvas-container');
    
    try {
        await showcase.init();
        
        // Hide loading
        const loader = document.getElementById('loading-overlay');
        if (loader) loader.style.opacity = '0';
        setTimeout(() => loader?.remove(), 500);
        
        // Init Observer for Scrolling
        initScrollObserver(showcase);
        
    } catch (e) {
        console.error('Failed to init showcase:', e);
        if (document.querySelector('.loader')) {
            document.querySelector('.loader')!.textContent = 'Error loading engine.';
        }
    }
});

function initScrollObserver(showcase: ShowcaseScene) {
    const sections = document.querySelectorAll('.section');
    
    const options = {
        root: document.querySelector('.scroll-container'),
        threshold: 0.5
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                
                // Switch Logic
                if (entry.target.id === 'section-welcome') {
                    showcase.switchToScene(1);
                } else if (entry.target.id === 'section-features') {
                    showcase.switchToScene(2);
                }
            } else {
                entry.target.classList.remove('active');
            }
        });
    }, options);
    
    sections.forEach(section => {
        observer.observe(section);
    });
}

