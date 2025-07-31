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

        // Status elements
        this.connectionStatus = document.getElementById('connection-status');
        this.connectionText = document.getElementById('connection-text');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.currentDay = document.getElementById('current-day');
        this.activePersonas = document.getElementById('active-personas');
        this.eventsProcessed = document.getElementById('events-processed');
        this.assessmentsCompleted = document.getElementById('assessments-completed');
        this.avgResponseTime = document.getElementById('avg-response-time');

        // Configuration elements
        this.conditionSelect = document.getElementById('condition-select');
        this.includeAssessments = document.getElementById('include-assessments');
        this.includeMechanistic = document.getElementById('include-mechanistic');
        this.includeEvents = document.getElementById('include-events');

        // Activity log
        this.activityLog = document.getElementById('activity-log');
        
        // Load event types dynamically
        this.loadExperimentalConditions();
    }
    
    async loadExperimentalConditions() {
        console.log('loadExperimentalConditions called - starting to fetch experimental conditions...');
        try {
            const response = await fetch('/api/v1/data/experimental-conditions');
            console.log('Response received:', response);
            console.log('Response status:', response.status);
            console.log('Response ok:', response.ok);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const experimentalConditions = await response.json();
            console.log('Experimental conditions parsed:', experimentalConditions);
            
            // Clear existing options
            this.conditionSelect.innerHTML = '';
            
            // Add experimental condition options
            console.log('Adding experimental condition options...');
            experimentalConditions.conditions.forEach(condition => {
                console.log('Adding option:', condition);
                const option = document.createElement('option');
                option.value = condition.value;
                option.textContent = condition.label;
                option.title = condition.description;
                this.conditionSelect.appendChild(option);
            });
            
            console.log('Final select options:', this.conditionSelect.innerHTML);
            console.log('Experimental conditions loaded successfully:', experimentalConditions);
        } catch (error) {
            console.error('Failed to load experimental conditions:', error);
            // Fallback to default options if API fails
            const fallbackOptions = [
                { value: 'control', label: 'Control Condition' },
                { value: 'stress', label: 'Stress Condition' },
                { value: 'neutral', label: 'Neutral Condition' },
                { value: 'minimal', label: 'Minimal Condition' }
            ];
            
            console.log('Using fallback options:', fallbackOptions);
            fallbackOptions.forEach(condition => {
                const option = document.createElement('option');
                option.value = condition.value;
                option.textContent = condition.label;
                this.conditionSelect.appendChild(option);
            });
        }
    }
    
    initializeCharts() {
        console.log('Initializing charts...');
        
        // Events Processed Chart
        const eventsCtx = document.getElementById('events-chart');
        console.log('Events chart canvas element:', eventsCtx);
        if (!eventsCtx) {
            console.error('Events chart canvas not found!');
            return;
        }
        
        this.charts.events = new Chart(eventsCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Events Processed',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
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
                        text: 'Events Processed Over Time'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Events'
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(0);
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
        console.log('Events chart initialized');
        
        // Assessment Completion Chart
        const assessmentsCtx = document.getElementById('assessments-chart');
        console.log('Assessments chart canvas element:', assessmentsCtx);
        if (!assessmentsCtx) {
            console.error('Assessments chart canvas not found!');
            return;
        }
        
        this.charts.assessments = new Chart(assessmentsCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Assessments Completed',
                    data: [],
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.1)',
                    tension: 0.4,
                    fill: true
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
                        text: 'Assessment Completion Rate'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Assessments'
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(0);
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
        console.log('Assessments chart initialized');
        
        // Response Time Chart
        const responseTimeCtx = document.getElementById('response-time-chart');
        console.log('Response time chart canvas element:', responseTimeCtx);
        if (!responseTimeCtx) {
            console.error('Response time chart canvas not found!');
            return;
        }
        
        this.charts.responseTime = new Chart(responseTimeCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average Response Time (ms)',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    fill: true
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
                        text: 'Response Time Performance'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Response Time (milliseconds)'
                        },
                        ticks: {
                            callback: function(value) {
                                if (value >= 1000) {
                                    return (value / 1000).toFixed(1) + 's';
                                } else if (value >= 1) {
                                    return value.toFixed(0) + 'ms';
                                } else {
                                    return (value * 1000).toFixed(0) + 'μs';
                                }
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
        console.log('Response time chart initialized');
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startSimulation());
        this.pauseBtn.addEventListener('click', () => this.pauseSimulation());
        this.resumeBtn.addEventListener('click', () => this.resumeSimulation());
        this.stopBtn.addEventListener('click', () => this.stopSimulation());
        this.exportBtn.addEventListener('click', () => this.exportData());
        
        // Configuration change listeners
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
        
        this.ws.onopen = (event) => {
            console.log('WebSocket connected successfully');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('online', 'Connected');
            this.addActivityLog('WebSocket connected');
            
            // Send a test message to verify connection
            try {
                this.ws.send(JSON.stringify({type: 'request_status'}));
                console.log('Sent initial status request');
            } catch (error) {
                console.error('Error sending initial status request:', error);
            }
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
        
        this.ws.onclose = (event) => {
            console.log('WebSocket disconnected', event.code, event.reason);
            this.isConnected = false;
            this.updateConnectionStatus('offline', 'Disconnected');
            this.addActivityLog('WebSocket disconnected');
            
            // Attempt to reconnect
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                setTimeout(() => this.connectWebSocket(), 2000);
            } else {
                console.log('Max reconnection attempts reached');
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('error', 'Connection Error');
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
        this.assessmentsCompleted.textContent = data.assessments_completed || 0;
        
        // Update response time display
        const avgResponseTime = data.average_response_time || 0;
        let responseTimeDisplay;
        if (avgResponseTime >= 1) {
            responseTimeDisplay = `${(avgResponseTime * 1000).toFixed(0)}ms`;
        } else if (avgResponseTime >= 0.001) {
            responseTimeDisplay = `${(avgResponseTime * 1000000).toFixed(0)}μs`;
        } else {
            responseTimeDisplay = `${(avgResponseTime * 1000000000).toFixed(0)}ns`;
        }
        this.avgResponseTime.textContent = responseTimeDisplay;
        
        // Update charts with new data
        this.updateCharts(data);
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
    }
    
    handleMechanisticUpdate(data) {
        this.addActivityLog(`Mechanistic analysis updated`, 'mechanistic');
        this.updateCharts(data);
    }
    
    handleError(data) {
        this.addActivityLog(`Error: ${data.error}`, 'error');
    }
    
    handleAlert(data) {
        this.addActivityLog(`Alert: ${data.message}`, 'alert');
    }
    
    updateCharts(data) {
        console.log('updateCharts called with data:', data);
        const timestamp = new Date().toLocaleTimeString();
        
        // Update events chart
        if (data.events_processed !== undefined) {
            console.log('Updating events chart with:', data.events_processed);
            this.updateEventsChart(data.events_processed, timestamp);
        }
        
        // Update assessments chart
        if (data.assessments_completed !== undefined) {
            console.log('Updating assessments chart with:', data.assessments_completed);
            this.updateAssessmentsChart(data.assessments_completed, timestamp);
        }
        
        // Update response time chart
        if (data.average_response_time !== undefined) {
            console.log('Updating response time chart with:', data.average_response_time);
            this.updateResponseTimeChart(data.average_response_time, timestamp);
        }
    }
    
    updateEventsChart(eventsCount, timestamp) {
        console.log('updateEventsChart called with:', eventsCount, timestamp);
        const chart = this.charts.events;
        chart.data.labels.push(timestamp);
        chart.data.datasets[0].data.push(eventsCount);
        
        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update();
        console.log('Events chart updated');
    }
    
    updateAssessmentsChart(assessmentsCount, timestamp) {
        console.log('updateAssessmentsChart called with:', assessmentsCount, timestamp);
        const chart = this.charts.assessments;
        chart.data.labels.push(timestamp);
        chart.data.datasets[0].data.push(assessmentsCount);
        
        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update();
        console.log('Assessments chart updated');
    }
    
    updateResponseTimeChart(responseTime, timestamp) {
        console.log('updateResponseTimeChart called with:', responseTime, timestamp);
        const chart = this.charts.responseTime;
        chart.data.labels.push(timestamp);
        // Convert seconds to milliseconds
        chart.data.datasets[0].data.push(responseTime * 1000);
        
        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update();
        console.log('Response time chart updated');
    }
    
    updateAssessmentChart(data) {
        console.log('updateAssessmentChart called with data:', data);
        // This method is now deprecated since we're not showing assessment scores anymore
        // But keeping it for backward compatibility
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
            
            if (response.ok) {
                const result = await response.json();
                console.log('Simulation started:', result);
                this.addActivityLog('Simulation started successfully');
            } else {
                const error = await response.text();
                console.error('Failed to start simulation:', error);
                this.addActivityLog(`Failed to start simulation: ${error}`, 'error');
            }
        } catch (error) {
            console.error('Error starting simulation:', error);
            this.addActivityLog(`Error starting simulation: ${error.message}`, 'error');
        }
    }
    
    async pauseSimulation() {
        try {
            const response = await fetch('/api/v1/simulation/pause', {
                method: 'POST'
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Simulation paused:', result);
                this.addActivityLog('Simulation paused');
            } else {
                const error = await response.text();
                console.error('Failed to pause simulation:', error);
                this.addActivityLog(`Failed to pause simulation: ${error}`, 'error');
            }
        } catch (error) {
            console.error('Error pausing simulation:', error);
            this.addActivityLog(`Error pausing simulation: ${error.message}`, 'error');
        }
    }
    
    async resumeSimulation() {
        try {
            const response = await fetch('/api/v1/simulation/resume', {
                method: 'POST'
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Simulation resumed:', result);
                this.addActivityLog('Simulation resumed');
            } else {
                const error = await response.text();
                console.error('Failed to resume simulation:', error);
                this.addActivityLog(`Failed to resume simulation: ${error}`, 'error');
            }
        } catch (error) {
            console.error('Error resuming simulation:', error);
            this.addActivityLog(`Error resuming simulation: ${error.message}`, 'error');
        }
    }
    
    async stopSimulation() {
        try {
            const response = await fetch('/api/v1/simulation/stop', {
                method: 'POST'
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Simulation stopped:', result);
                this.addActivityLog('Simulation stopped');
            } else {
                const error = await response.text();
                console.error('Failed to stop simulation:', error);
                this.addActivityLog(`Failed to stop simulation: ${error}`, 'error');
            }
        } catch (error) {
            console.error('Error stopping simulation:', error);
            this.addActivityLog(`Error stopping simulation: ${error.message}`, 'error');
        }
    }
    
    async exportData() {
        try {
            const includeAssessments = this.includeAssessments.checked;
            const includeMechanistic = this.includeMechanistic.checked;
            const includeEvents = this.includeEvents.checked;
            
            const response = await fetch('/api/v1/simulation/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    include_assessments: includeAssessments,
                    include_mechanistic: includeMechanistic,
                    include_events: includeEvents
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Data exported:', result);
                this.addActivityLog('Data exported successfully');
                
                // Create download link if file path is provided
                if (result.file_path) {
                    const link = document.createElement('a');
                    link.href = `/api/v1/simulation/download/${result.simulation_id}`;
                    link.download = `simulation_data_${result.simulation_id}.zip`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }
            } else {
                const error = await response.text();
                console.error('Failed to export data:', error);
                this.addActivityLog(`Failed to export data: ${error}`, 'error');
            }
        } catch (error) {
            console.error('Error exporting data:', error);
            this.addActivityLog(`Error exporting data: ${error.message}`, 'error');
        }
    }
    
    updateConfiguration() {
        console.log('Configuration updated');
        this.addActivityLog('Configuration updated');
    }
    
    startPolling() {
        // Poll for status updates every 5 seconds
        this.pollingInterval = setInterval(async () => {
            if (!this.isConnected) {
                return;
            }
            
            try {
                const response = await fetch('/api/v1/simulation/status');
                if (response.ok) {
                    const status = await response.json();
                    this.updateSimulationStatus(status);
                }
            } catch (error) {
                console.error('Error polling status:', error);
            }
        }, 5000);
    }
    
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SimulationDashboard();
}); 