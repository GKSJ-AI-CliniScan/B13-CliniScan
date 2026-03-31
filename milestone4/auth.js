document.addEventListener('DOMContentLoaded', () => {
    const currentUser = JSON.parse(localStorage.getItem('cliniscanCurrentUser'));
    
    // Pages that require authentication
    const currentPath = window.location.pathname.split('/').pop();
    const isProtected = ['index.html', 'dashboard.html', ''].includes(currentPath);

    if (isProtected && !currentUser) {
        window.location.href = 'login.html';
        return;
    }

    // Update Navigation UI
    const navMenu = document.querySelector('.navbar');
    if (navMenu && currentUser) {
        // Remove old commented profile if exists but we'll just append our own
        let profileSection = document.querySelector('.nav-profile');
        if (!profileSection) {
            profileSection = document.createElement('div');
            profileSection.className = 'nav-profile';
            profileSection.style.display = 'flex';
            profileSection.style.alignItems = 'center';
            profileSection.style.gap = '15px';
            profileSection.style.marginLeft = 'auto'; // push to the right
            navMenu.appendChild(profileSection);
        }
        
        profileSection.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <img src="https://ui-avatars.com/api/?name=${encodeURIComponent(currentUser.name)}&background=0d6efd&color=fff&rounded=true" alt="Profile" style="width: 32px; height: 32px; border-radius: 50%;">
                <span style="font-weight: 500; font-size: 0.9rem; color: var(--text-main);">${currentUser.name}</span>
            </div>
            <button id="logoutBtn" style="background: none; border: 1px solid #ef4444; color: #ef4444; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-family: inherit; font-size: 0.85rem; font-weight: 500; transition: all 0.2s;">Logout</button>
        `;

        document.getElementById('logoutBtn').addEventListener('click', () => {
            localStorage.removeItem('cliniscanCurrentUser');
            window.location.href = 'login.html';
        });
        
        // Add hover effect via JS since it's inline styled
        const logoutBtn = document.getElementById('logoutBtn');
        logoutBtn.addEventListener('mouseenter', () => {
            logoutBtn.style.background = '#fef2f2';
        });
        logoutBtn.addEventListener('mouseleave', () => {
            logoutBtn.style.background = 'none';
        });
    } else if (navMenu && !currentUser && currentPath !== 'login.html' && currentPath !== 'register.html') {
        const loginLink = document.createElement('div');
        loginLink.className = 'nav-profile';
        loginLink.style.marginLeft = 'auto';
        loginLink.innerHTML = `<a href="login.html" class="btn-primary" style="text-decoration: none; padding: 8px 16px; border-radius: 6px; font-size: 0.9rem;">Login</a>`;
        navMenu.appendChild(loginLink);
    }
});
