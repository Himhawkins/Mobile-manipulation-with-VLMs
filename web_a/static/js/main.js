// /dashboard_webapp/static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const cameraMenu = document.getElementById('camera-menu');
    const serialMenu = document.getElementById('serial-menu');
    const modeMenu = document.getElementById('mode-menu');
    const calibrateBtn = document.getElementById('calibrate-btn');
    const runBtn = document.getElementById('run-btn');
    const executeBtn = document.getElementById('execute-btn');
    const refreshCamerasBtn = document.getElementById('refresh-cameras-btn');
    const videoFeed = document.getElementById('video-feed');
    const inputValue = document.getElementById('input-entry');

    // --- Helper Functions ---
    const populateSelect = (selectElement, options) => {
        selectElement.innerHTML = '';
        if (!options || options.length === 0) {
            selectElement.innerHTML = '<option>None found</option>';
            return;
        }
        options.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            selectElement.appendChild(opt);
        });
    };
    
    // --- API Calls ---
    const fetchInitialData = async () => {
        try {
            const response = await fetch('/api/refresh_sources');
            const data = await response.json();
            populateSelect(cameraMenu, data.cameras);
            populateSelect(serialMenu, data.ports);
            populateSelect(modeMenu, data.agents);
        } catch (error) {
            console.error('Failed to fetch initial data:', error);
        }
    };

    const setCamera = async () => {
        const selectedOption = cameraMenu.options[cameraMenu.selectedIndex].value;
        const camIndex = selectedOption.match(/\d+/)[0]; // Extracts number from "Camera 0"
        
        console.log(`Setting camera to index ${camIndex}`);
        await fetch('/api/set_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_index: camIndex }),
        });
        
        // Force reload the image source to reflect the new camera feed
        videoFeed.src = `/video_feed?t=${new Date().getTime()}`;
    };
    
    const runAction = async (button, action) => {
        button.disabled = true;
        try {
            const response = await fetch('/api/run_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: action,
                    inputValue: inputValue.value,
                    modeValue: modeMenu.value
                })
            });
            const result = await response.json();
            console.log(result.output); // Log output to console
            // You could update a text area with this output
            document.getElementById('preview1-text').textContent = result.output;
        } catch (error) {
            console.error(`Action ${action} failed:`, error);
        } finally {
            button.disabled = false;
        }
    };

    // --- Event Listeners ---
    cameraMenu.addEventListener('change', setCamera);
    refreshCamerasBtn.addEventListener('click', fetchInitialData); // Refreshes everything for simplicity
    
    calibrateBtn.addEventListener('click', () => runAction(calibrateBtn, 'Calibrate'));
    runBtn.addEventListener('click', () => runAction(runBtn, 'Run'));

    executeBtn.addEventListener('click', () => {
        const isRunning = executeBtn.classList.toggle('running');
        executeBtn.textContent = isRunning ? 'Stop' : 'Execute';
        // Add API call to '/api/toggle_execute' here
        console.log(`Execution state: ${isRunning ? 'started' : 'stopped'}`);
    });

    // --- Initial Load ---
    fetchInitialData();
});
