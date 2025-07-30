// Dashboard Application
class SimulationDashboard {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.charts = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.pollingInterval = null;
        
        this.initializeElements();
        this.initializeCharts();
        this.initializeEventListeners();
        this.connectWebSocket();
        this.startPolling();
    }
    
    initializeElements() {
        // Control buttons
        this.startBtn = document.getElementById('start-btn');
        this.pauseBtn = document.getElementById('pause-btn');
        this.resumeBtn = document.getElementById('resume-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.exportBtn = document.getElementById('export-btn');
        this.testChartsBtn = document.getElementById('test-charts-btn');

        // Status elements
        this.connectionStatus = document.getElementById('connection-status');
        this.connectionText = document.getElementById('connection-text');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.currentDay = document.getElementById('current-day');
        this.activePersonas = document.getElementById('active-personas');
        this.eventsProcessed = document.getElementById('events-processed');

        // Configuration elements
        this.conditionSelect = document.getElementById('condition-select');
        this.includeAssessments = document.getElementById('include-assessments');
        this.includeMechanistic = document.getElementById('include-mechanistic');
        this.includeEvents = document.getElementById('include-events');

        // Activity log
        this.activityLog = document.getElementById('activity-log');
    }
    
    initializeCharts() {
        console.log('Initializing charts...');
        
        // Personality Drift Chart
        const driftCtx = document.getElementById('drift-chart');
        console.log('Drift chart canvas element:', driftCtx);
        if (!driftCtx) {
            console.error('Drift chart canvas not found!');
            return;
        }
        
        this.charts.drift = new Chart(driftCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Openness',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Conscientiousness',
                    data: [],
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Neuroticism',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Personality Trait Changes Over Time'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        console.log('Drift chart initialized');
        
        // Assessment Scores Chart
        const assessmentCtx = document.getElementById('assessment-chart');
        console.log('Assessment chart canvas element:', assessmentCtx);
        if (!assessmentCtx) {
            console.error('Assessment chart canvas not found!');
            return;
        }
        
        this.charts.assessment = new Chart(assessmentCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['PHQ-9', 'GAD-7', 'PSS-10'],
                datasets: [{
                    label: 'Current Score',
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(39, 174, 96, 0.8)',
                        'rgba(231, 76, 60, 0.8)'
                    ],
                    borderColor: [
                        'rgba(102, 126, 234, 1)',
                        'rgba(39, 174, 96, 1)',
                        'rgba(231, 76, 60, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Assessment Scores'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        console.log('Assessment chart initialized');
    }
    
    initializeEventListeners() {
        // Control buttons
        this.startBtn.addEventListener('click', () => this.startSimulation());
        this.pauseBtn.addEventListener('click', () => this.pauseSimulation());
        this.resumeBtn.addEventListener('click', () => this.resumeSimulation());
        this.stopBtn.addEventListener('click', () => this.stopSimulation());
        this.exportBtn.addEventListener('click', () => this.exportData());
        this.testChartsBtn.addEventListener('click', () => this.testCharts());
        
        // Configuration changes
        this.conditionSelect.addEventListener('change', () => this.updateConfiguration());
        this.includeAssessments.addEventListener('change', () => this.updateConfiguration());
        this.includeMechanistic.addEventListener('change', () => this.updateConfiguration());
        this.includeEvents.addEventListener('change', () => this.updateConfiguration());
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/simulation`;
        
        console.log('Connecting to WebSocket:', wsUrl);
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected successfully');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('online', 'Connected');
            this.addActivityLog('WebSocket connected');
        };
        
        this.ws.onmessage = (event) => {
            console.log('Raw WebSocket message received:', event.data);
            try {
                const message = JSON.parse(event.data);
                console.log('Parsed WebSocket message:', message);
                this.handleWebSocketMessage(message);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
                console.error('Raw message was:', event.data);
            }
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('offline', 'Disconnected');
            this.addActivityLog('WebSocket disconnected');
            
            // Attempt to reconnect
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                setTimeout(() => this.connectWebSocket(), 2000);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('offline', 'Connection Error');
        };
    }
    
    handleWebSocketMessage(message) {
        console.log('Received WebSocket message:', message);
        
        // Add a simple counter to track message types
        if (!this.messageCounts) {
            this.messageCounts = {};
        }
        this.messageCounts[message.type] = (this.messageCounts[message.type] || 0) + 1;
        console.log('Message counts:', this.messageCounts);
        
        switch (message.type) {
            case 'simulation_status':
                console.log('Processing simulation status:', message.data);
                this.updateSimulationStatus(message.data);
                break;
            case 'progress_update':
                console.log('Processing progress update:', message.data);
                this.updateProgress(message.data);
                break;
            case 'event_occurred':
                console.log('Processing event occurred:', message.data);
                this.handleEventOccurred(message.data);
                break;
            case 'assessment_completed':
                console.log('Processing assessment completed:', message.data);
                this.handleAssessmentCompleted(message.data);
                break;
            case 'mechanistic_update':
                console.log('Processing mechanistic update:', message.data);
                this.handleMechanisticUpdate(message.data);
                break;
            case 'error':
                console.log('Processing error:', message.data);
                this.handleError(message.data);
                break;
            case 'alert':
                console.log('Processing alert:', message.data);
                this.handleAlert(message.data);
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    updateConnectionStatus(status, text) {
        this.connectionStatus.className = `status-dot ${status}`;
        this.connectionText.textContent = text;
    }
    
    updateSimulationStatus(data) {
        if (data.is_running) {
            this.startBtn.disabled = true;
            this.pauseBtn.disabled = false;
            this.resumeBtn.disabled = true;
            this.stopBtn.disabled = false;
        } else if (data.is_paused) {
            this.startBtn.disabled = true;
            this.pauseBtn.disabled = true;
            this.resumeBtn.disabled = false;
            this.stopBtn.disabled = false;
        } else {
            this.startBtn.disabled = false;
            this.pauseBtn.disabled = true;
            this.resumeBtn.disabled = true;
            this.stopBtn.disabled = true;
        }
        
        this.updateProgress(data);
    }
    
    updateProgress(data) {
        const progress = data.progress_percentage || 0;
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = `${progress.toFixed(1)}% Complete`;
        
        this.currentDay.textContent = `${data.current_day || 0} / ${data.total_days || 30}`;
        this.activePersonas.textContent = data.active_personas || 0;
        this.eventsProcessed.textContent = data.events_processed || 0;
    }
    
    handleEventOccurred(data) {
        this.addActivityLog(`Event occurred: ${data.title}`, 'event');
        this.updateCharts(data);
    }
    
    handleAssessmentCompleted(data) {
        console.log('Processing assessment completed:', data);
        this.addActivityLog(`Assessment completed: ${data.assessment_type}`, 'assessment');
        
        // Update assessment chart
        this.updateAssessmentChart(data);
        
        // Also update drift chart with personality data from assessment
        if (data.personality_traits) {
            console.log('Updating drift chart with personality data:', data.personality_traits);
            this.updateCharts(data);
        } else {
            console.log('No personality traits found in assessment data');
        }
    }
    
    handleMechanisticUpdate(data) {
        this.addActivityLog(`Mechanistic analysis updated`, 'mechanistic');
        this.updateDriftChart(data);
    }
    
    handleError(data) {
        this.addActivityLog(`Error: ${data.error}`, 'error');
    }
    
    handleAlert(data) {
        this.addActivityLog(`Alert: ${data.message}`, 'alert');
    }
    
    updateCharts(data) {
        console.log('updateCharts called with data:', data);
        // Update drift chart with new personality data
        if (data.personality_traits) {
            console.log('Personality traits found:', data.personality_traits);
            const labels = this.charts.drift.data.labels;
            const datasets = this.charts.drift.data.datasets;
            
            // Add new time point
            labels.push(new Date().toLocaleTimeString());
            
            // Update datasets with new trait values
            datasets[0].data.push(data.personality_traits.openness || 0.5);
            datasets[1].data.push(data.personality_traits.conscientiousness || 0.5);
            datasets[2].data.push(data.personality_traits.neuroticism || 0.5);
            
            // Keep only last 20 data points
            if (labels.length > 20) {
                labels.shift();
                datasets.forEach(dataset => dataset.data.shift());
            }
            
            console.log('Updating drift chart with data points:', datasets[0].data.length);
            this.charts.drift.update();
        } else {
            console.log('No personality_traits found in data');
        }
    }
    
    updateAssessmentChart(data) {
        console.log('updateAssessmentChart called with data:', data);
        const chart = this.charts.assessment;
        
        // Handle the data structure that backend actually sends
        if (data.phq9_score !== undefined || data.gad7_score !== undefined || data.pss10_score !== undefined) {
            console.log('Using direct score fields:', { phq9: data.phq9_score, gad7: data.gad7_score, pss10: data.pss10_score });
            chart.data.datasets[0].data = [
                data.phq9_score || 0,
                data.gad7_score || 0,
                data.pss10_score || 0
            ];
            chart.update();
        }
        // Also handle the old structure for backward compatibility
        else if (data.scores) {
            console.log('Using scores object:', data.scores);
            chart.data.datasets[0].data = [
                data.scores.phq9 || 0,
                data.scores.gad7 || 0,
                data.scores.pss10 || 0
            ];
            chart.update();
        } else {
            console.log('No assessment scores found in data');
        }
    }
    
    updateDriftChart(data) {
        // Update drift indicators
        if (data.drift_indicators) {
            const labels = this.charts.drift.data.labels;
            const datasets = this.charts.drift.data.datasets;
            
            labels.push(new Date().toLocaleTimeString());
            
            // Add drift data
            datasets[0].data.push(data.drift_indicators.openness_drift || 0.5);
            datasets[1].data.push(data.drift_indicators.conscientiousness_drift || 0.5);
            datasets[2].data.push(data.drift_indicators.neuroticism_drift || 0.5);
            
            if (labels.length > 20) {
                labels.shift();
                datasets.forEach(dataset => dataset.data.shift());
            }
            
            this.charts.drift.update();
        }
    }
    
    addActivityLog(message, type = 'info') {
        const timestamp = new Date().toLocaleString();
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        
        const timestampSpan = document.createElement('span');
        timestampSpan.className = 'timestamp';
        timestampSpan.textContent = timestamp;
        
        const messageSpan = document.createElement('span');
        messageSpan.className = 'message';
        messageSpan.textContent = message;
        
        // Add color coding for different types
        if (type === 'error') {
            messageSpan.style.color = '#e74c3c';
        } else if (type === 'event') {
            messageSpan.style.color = '#f39c12';
        } else if (type === 'assessment') {
            messageSpan.style.color = '#27ae60';
        } else if (type === 'mechanistic') {
            messageSpan.style.color = '#9b59b6';
        }
        
        activityItem.appendChild(timestampSpan);
        activityItem.appendChild(messageSpan);
        
        this.activityLog.appendChild(activityItem);
        
        // Keep only last 50 items
        while (this.activityLog.children.length > 50) {
            this.activityLog.removeChild(this.activityLog.firstChild);
        }
        
        // Auto-scroll to bottom
        this.activityLog.scrollTop = this.activityLog.scrollHeight;
    }
    
    async startSimulation() {
        try {
            const condition = this.conditionSelect.value;
            const response = await fetch('/api/v1/simulation/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    experimental_condition: condition,
                    config_name: 'experimental_design'
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.addActivityLog(`Simulation started: ${condition} condition`);
            } else {
                this.addActivityLog(`Failed to start simulation: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.addActivityLog(`Error starting simulation: ${error.message}`, 'error');
        }
    }
    
    async pauseSimulation() {
        try {
            const response = await fetch('/api/v1/simulation/pause', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.addActivityLog('Simulation paused');
            } else {
                this.addActivityLog(`Failed to pause simulation: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.addActivityLog(`Error pausing simulation: ${error.message}`, 'error');
        }
    }
    
    async resumeSimulation() {
        try {
            const response = await fetch('/api/v1/simulation/resume', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.addActivityLog('Simulation resumed');
            } else {
                this.addActivityLog(`Failed to resume simulation: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.addActivityLog(`Error resuming simulation: ${error.message}`, 'error');
        }
    }
    
    async stopSimulation() {
        try {
            const response = await fetch('/api/v1/simulation/stop', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.addActivityLog('Simulation stopped');
            } else {
                this.addActivityLog(`Failed to stop simulation: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.addActivityLog(`Error stopping simulation: ${error.message}`, 'error');
        }
    }
    
    async exportData() {
        try {
            const response = await fetch('/api/v1/data/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    export_format: 'json',
                    include_assessments: this.includeAssessments.checked,
                    include_mechanistic: this.includeMechanistic.checked,
                    include_events: this.includeEvents.checked
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.addActivityLog(`Data exported: ${result.filename}`);
                
                // Trigger download
                const downloadUrl = `/api/v1/data/download/${result.filename}`;
                const link = document.createElement('a');
                link.href = downloadUrl;
                link.download = result.filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            } else {
                this.addActivityLog(`Failed to export data: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.addActivityLog(`Error exporting data: ${error.message}`, 'error');
        }
    }
    
    updateConfiguration() {
        const condition = this.conditionSelect.value;
        this.addActivityLog(`Configuration updated: ${condition} condition`);
    }

    startPolling() {
        // Poll for simulation status every 2 seconds
        this.pollingInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/v1/simulation/status');
                if (response.ok) {
                    const status = await response.json();
                    console.log('Polled status:', status);
                    this.updateSimulationStatus(status);
                }
            } catch (error) {
                console.error('Error polling status:', error);
            }
        }, 2000);
    }
    
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    testCharts() {
        console.log('Testing charts...');
        
        // Test drift chart
        const testData = {
            personality_traits: {
                openness: 0.7,
                conscientiousness: 0.8,
                neuroticism: 0.3
            }
        };
        
        console.log('Testing drift chart with data:', testData);
        this.updateCharts(testData);
        
        // Test assessment chart
        const testAssessmentData = {
            phq9_score: 5,
            gad7_score: 3,
            pss10_score: 12
        };
        
        console.log('Testing assessment chart with data:', testAssessmentData);
        this.updateAssessmentChart(testAssessmentData);
        
        this.addActivityLog('Charts tested with sample data', 'info');
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SimulationDashboard();
}); 