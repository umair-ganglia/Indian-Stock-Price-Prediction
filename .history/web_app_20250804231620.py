<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indian Stock Price Prediction - AI Powered</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --accent-color: #f093fb;
            --success-color: #4facfe;
            --warning-color: #fa709a;
            --error-color: #ff6b6b;
            --info-color: #4ecdc4;
            
            /* Light theme */
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #e2e8f0;
            --text-primary: #1a202c;
            --text-secondary: #4a5568;
            --text-muted: #718096;
            --border-color: #e2e8f0;
            --shadow-light: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-medium: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-heavy: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }

        [data-theme="dark"] {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-color: #475569;
            --shadow-light: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            --shadow-medium: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            --shadow-heavy: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            transition: all 0.3s ease;
        }

        .app-container {
            display: flex;
            min-height: 100vh;
        }

        /* Enhanced Sidebar */
        .sidebar {
            width: 350px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--border-color);
            transition: all 0.3s ease;
            overflow-y: auto;
            position: relative;
        }

        .sidebar.collapsed {
            width: 80px;
        }

        .sidebar-header {
            padding: 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: white;
            font-size: 1.25rem;
            font-weight: bold;
        }

        .collapse-btn {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: white;
            padding: 0.5rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .collapse-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.1);
        }

        .sidebar-content {
            padding: 1.5rem;
            color: white;
        }

        .sidebar.collapsed .sidebar-content {
            display: none;
        }

        .section {
            margin-bottom: 2rem;
        }

        .section-title {
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            opacity: 0.8;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-label {
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: rgba(255, 255, 255, 0.9);
        }

        .form-input, .form-select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 0.875rem;
            transition: all 0.3s ease;
        }

        .form-input:focus, .form-select:focus {
            outline: none;
            border-color: var(--accent-color);
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 0 3px rgba(240, 147, 251, 0.1);
        }

        .form-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }

        .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .checkbox-item:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(4px);
        }

        .checkbox-item input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: var(--accent-color);
        }

        .slider-container {
            margin: 1rem 0;
        }

        .slider {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.2);
            outline: none;
            opacity: 0.7;
            transition: opacity 0.2s;
        }

        .slider:hover {
            opacity: 1;
        }

        .slider::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent-color);
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }

        .btn {
            padding: 0.875rem 1.5rem;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            justify-content: center;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-color), #f093fb);
            color: white;
            box-shadow: var(--shadow-medium);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-heavy);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .btn-group {
            display: flex;
            gap: 0.75rem;
            margin-top: 1.5rem;
        }

        .btn-group .btn {
            flex: 1;
        }

        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .top-bar {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-light);
        }

        .theme-toggle {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .theme-toggle:hover {
            background: var(--primary-color);
            color: white;
            transform: scale(1.05);
        }

        .content-area {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
            background: var(--bg-primary);
        }

        .header {
            text-align: center;
            margin-bottom: 3rem;
        }

        .main-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            line-height: 1.2;
        }

        .subtitle {
            font-size: 1.25rem;
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }

        .tabs {
            display: flex;
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 0.5rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-light);
        }

        .tab {
            flex: 1;
            padding: 0.875rem 1rem;
            text-align: center;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .tab.active {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            box-shadow: var(--shadow-medium);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .welcome-screen {
            text-align: center;
            padding: 3rem 0;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
        }

        .feature-card {
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: var(--shadow-medium);
            transition: all 0.3s ease;
            border: 1px solid var(--border-color);
        }

        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-heavy);
        }

        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
        }

        .feature-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }

        .feature-description {
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .metric-card {
            background: linear-gradient(135deg, var(--success-color), #00f2fe);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            box-shadow: var(--shadow-medium);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-heavy);
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            font-size: 0.875rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .analysis-section {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: var(--shadow-medium);
            border: 1px solid var(--border-color);
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .section-header h3 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .chart-container {
            background: var(--bg-primary);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: var(--shadow-light);
            border: 1px solid var(--border-color);
        }

        .progress-container {
            margin: 2rem 0;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .status-message {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 500;
        }

        .status-success {
            background: rgba(79, 172, 254, 0.1);
            color: var(--success-color);
            border: 1px solid rgba(79, 172, 254, 0.3);
        }

        .status-warning {
            background: rgba(247, 112, 154, 0.1);
            color: var(--warning-color);
            border: 1px solid rgba(247, 112, 154, 0.3);
        }

        .status-info {
            background: rgba(78, 205, 196, 0.1);
            color: var(--info-color);
            border: 1px solid rgba(78, 205, 196, 0.3);
        }

        .alert {
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid;
        }

        .alert-success {
            background: rgba(79, 172, 254, 0.1);
            border-left-color: var(--success-color);
            color: var(--success-color);
        }

        .alert-warning {
            background: rgba(247, 112, 154, 0.1);
            border-left-color: var(--warning-color);
            color: var(--warning-color);
        }

        .alert-info {
            background: rgba(78, 205, 196, 0.1);
            border-left-color: var(--info-color);
            color: var(--info-color);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                position: fixed;
                top: 0;
                left: -100%;
                z-index: 1000;
                height: 100vh;
                transition: left 0.3s ease;
            }

            .sidebar.open {
                left: 0;
            }

            .main-content {
                width: 100%;
            }

            .main-title {
                font-size: 2.5rem;
            }

            .feature-grid {
                grid-template-columns: 1fr;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .top-bar {
                padding: 1rem;
            }

            .content-area {
                padding: 1rem;
            }
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }

        .slide-in {
            animation: slideIn 0.6s ease-out;
        }

        /* Loading States */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .pulse {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body data-theme="light">
    <div class="app-container">
        <!-- Enhanced Sidebar -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <i class="fas fa-chart-line"></i>
                    <span class="logo-text">StockAI</span>
                </div>
                <button class="collapse-btn" onclick="toggleSidebar()">
                    <i class="fas fa-bars"></i>
                </button>
            </div>

            <div class="sidebar-content">
                <!-- Stock Selection Section -->
                <div class="section">
                    <div class="section-title">
                        <i class="fas fa-search"></i>
                        Stock Selection
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Search Stock</label>
                        <input type="text" class="form-input" id="stockSearch" placeholder="e.g., TCS, Reliance, HDFC" oninput="filterStocks()">
                    </div>

                    <div class="form-group">
                        <label class="form-label">Select Stock Symbol</label>
                        <select class="form-select" id="stockSelect">
                            <option value="TCS.NS">TCS.NS - Tata Consultancy Services</option>
                            <option value="INFY.NS">INFY.NS - Infosys Limited</option>
                            <option value="RELIANCE.NS">RELIANCE.NS - Reliance Industries</option>
                            <option value="HDFCBANK.NS">HDFCBANK.NS - HDFC Bank</option>
                            <option value="ICICIBANK.NS">ICICIBANK.NS - ICICI Bank</option>
                            <option value="SBIN.NS">SBIN.NS - State Bank of India</option>
                            <option value="WIPRO.NS">WIPRO.NS - Wipro Limited</option>
                            <option value="TECHM.NS">TECHM.NS - Tech Mahindra</option>
                            <option value="MARUTI.NS">MARUTI.NS - Maruti Suzuki</option>
                            <option value="HINDUNILVR.NS">HINDUNILVR.NS - Hindustan Unilever</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Custom Stock Symbol</label>
                        <input type="text" class="form-input" id="customStock" placeholder="Enter custom symbol (e.g., GOOGL, AAPL)">
                    </div>
                </div>

                <!-- Time Period Section -->
                <div class="section">
                    <div class="section-title">
                        <i class="fas fa-calendar"></i>
                        Time Period
                    </div>
                    
                    <div class="form-group">
                        <select class="form-select" id="timePeriod">
                            <option value="3mo">3 Months (Short Term)</option>
                            <option value="6mo">6 Months (Medium Term)</option>
                            <option value="1y" selected>1 Year (Recommended)</option>
                            <option value="2y">2 Years (Long Term)</option>
                            <option value="5y">5 Years (Very Long Term)</option>
                        </select>
                    </div>
                </div>

                <!-- AI Models Section -->
                <div class="section">
                    <div class="section-title">
                        <i class="fas fa-robot"></i>
                        AI Models
                    </div>
                    
                    <div class="checkbox-group">
                        <label class="checkbox-item">
                            <input type="checkbox" id="linearModel" checked>
                            <div>
                                <strong>📊 Linear Regression</strong>
                                <small style="display: block; opacity: 0.8;">Fast and interpretable</small>
                            </div>
                        </label>
                        
                        <label class="checkbox-item">
                            <input type="checkbox" id="lstmModel" checked>
                            <div>
                                <strong>🧠 LSTM Neural Network</strong>
                                <small style="display: block; opacity: 0.8;">Deep learning model</small>
                            </div>
                        </label>
                        
                        <label class="checkbox-item">
                            <input type="checkbox" id="prophetModel" checked>
                            <div>
                                <strong>📈 Prophet Time Series</strong>
                                <small style="display: block; opacity: 0.8;">Handles seasonality well</small>
                            </div>
                        </label>
                    </div>
                </div>

                <!-- Advanced Settings -->
                <div class="section">
                    <div class="section-title">
                        <i class="fas fa-cog"></i>
                        Advanced Settings
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Days to Predict: <span id="predictionDaysValue">30</span></label>
                        <input type="range" class="slider" id="predictionDays" min="7" max="90" value="30" oninput="updateSliderValue('predictionDays', 'predictionDaysValue')">
                    </div>

                    <div class="form-group">
                        <label class="form-label">LSTM Training Epochs: <span id="lstmEpochsValue">50</span></label>
                        <input type="range" class="slider" id="lstmEpochs" min="10" max="100" value="50" oninput="updateSliderValue('lstmEpochs', 'lstmEpochsValue')">
                    </div>

                    <div class="form-group">
                        <label class="form-label">Risk Tolerance</label>
                        <select class="form-select" id="riskTolerance">
                            <option value="Conservative">Conservative</option>
                            <option value="Moderate" selected>Moderate</option>
                            <option value="Aggressive">Aggressive</option>
                        </select>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="startAnalysis()">
                        <i class="fas fa-rocket"></i>
                        Start Analysis
                    </button>
                </div>
                
                <div class="btn-group">
                    <button class="btn btn-secondary" onclick="clearCache()">
                        <i class="fas fa-trash"></i>
                        Clear Cache
                    </button>
                </div>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Top Bar -->
            <div class="top-bar">
                <button class="theme-toggle" onclick="toggleTheme()">
                    <i class="fas fa-moon" id="themeIcon"></i>
                    <span id="themeText">Dark Mode</span>
                </button>
                
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="color: var(--text-secondary);">Last Updated: <span id="lastUpdated">Never</span></span>
                    <button class="btn btn-secondary" onclick="refreshData()" style="padding: 0.5rem 1rem;">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                </div>
            </div>

            <!-- Content Area -->
            <div class="content-area">
                <!-- Header -->
                <div class="header fade-in">
                    <h1 class="main-title">📈 Indian Stock Price Prediction</h1>
                    <p class="subtitle">AI-Powered Stock Analysis & Future Price Prediction</p>
                </div>

                <!-- Navigation Tabs -->
                <div class="tabs">
                    <div class="tab active" onclick="switchTab('analysis')">
                        <i class="fas fa-chart-line"></i> Stock Analysis
                    </div>
                    <div class="tab" onclick="switchTab('portfolio')">
                        <i class="fas fa-briefcase"></i> Portfolio
                    </div>
                    <div class="tab" onclick="switchTab('realtime')">
                        <i class="fas fa-bolt"></i> Real-Time
                    </div>
                    <div class="tab" onclick="switchTab('insights')">
                        <i class="fas fa-brain"></i> AI Insights
                    </div>
                    <div class="tab" onclick="switchTab('scanner')">
                        <i class="fas fa-search"></i> Market Scanner
                    </div>
                </div>

                <!-- Tab Contents -->
                <div id="analysisTab" class="tab-content active">
                    <div class="welcome-screen" id="welcomeScreen">
                        <div class="alert alert-success fade-in">
                            <h3>🎯 Welcome to Advanced Stock Prediction!</h3>
                            <p>Harness the power of AI to predict Indian stock prices with multiple machine learning models.</p>
                        </div>

                        <!-- Quick Stats -->
                        <div class="metrics-grid fade-in">
                            <div class="metric-card">
                                <div class="metric-value">40+</div>
                                <div class="metric-label">Available Stocks</div>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, var(--warning-color), #fee140);">
                                <div class="metric-value">3</div>
                                <div class="metric-label">AI Models</div>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, var(--info-color), #44a08d);">
                                <div class="metric-value">15+</div>
                                <div class="metric-label">Sectors Covered</div>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, var(--error-color), #ffa726);">
                                <div class="metric-value">&lt; 60s</div>
                                <div class="metric-label">Prediction Speed</div>
                            </div>
                        </div>

                        <!-- Features -->
                        <div class="feature-grid fade-in">
                            <div class="feature-card slide-in">
                                <span class="feature-icon">📊</span>
                                <h3 class="feature-title">Linear Regression</h3>
                                <p class="feature-description">Traditional statistical approach for trend analysis with high interpretability and fast execution.</p>
                            </div>
                            <div class="feature-card slide-in">
                                <span class="feature-icon">🧠</span>
                                <h3 class="feature-title">LSTM Networks</h3>
                                <p class="feature-description">Deep learning for complex pattern recognition and sequential data analysis.</p>
                            </div>
                            <div class="feature-card slide-in">
                                <span class="feature-icon">📈</span>
                                <h3 class="feature-title">Prophet</h3>
                                <p class="feature-description">Facebook's time-series forecasting with seasonality handling and trend analysis.</p>
                            </div>
                        </div>

                        <!-- Getting Started Guide -->
                        <div class="analysis-section fade-in">
                            <div class="section-header">
                                <i class="fas fa-rocket"></i>
                                <h3>Getting Started Guide</h3>
                            </div>
                            <div class="alert alert-info">
                                <h4>📋 Quick Start Steps</h4>
                                <ol style="margin: 1rem 0; padding-left: 1.5rem;">
                                    <li><strong>Select a Stock:</strong> Choose from 40+ Indian stocks or enter a custom symbol</li>
                                    <li><strong>Pick Time Period:</strong> Select from 3 months to 5 years of historical data</li>
                                    <li><strong>Choose Models:</strong> Enable the AI models you want to use</li>
                                    <li><strong>Configure Settings:</strong> Adjust prediction days and model parameters</li>
                                    <li><strong>Start Analysis:</strong> Click the "Start Analysis" button and wait for results</li>
                                </ol>
                                <p><strong>💡 Tip:</strong> For best results, use 1-2 years of data with all three models enabled!</p>
                            </div>
                        </div>
                    </div>

                    <!-- Analysis Results (Hidden by default) -->
                    <div id="analysisResults" style="display: none;">
                        <!-- Progress Section -->
                        <div class="progress-container">
                            <div class="status-message status-info" id="statusMessage">
                                <i class="fas fa-info-circle"></i>
                                <span>Ready to start analysis...</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                            </div>
                        </div>

                        <!-- Stock Overview -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <i class="fas fa-chart-area"></i>
                                <h3>Stock Overview</h3>
                            </div>
                            
                            <div class="metrics-grid">
                                <div class="metric-card">
                                    <div class="metric-value" id="currentPrice">₹0.00</div>
                                    <div class="metric-label">Current Price</div>
                                    <div style="font-size: 0.875rem; margin-top: 0.5rem;" id="priceChange">+0.00 (0.00%)</div>
                                </div>
                                <div class="metric-card" style="background: linear-gradient(135deg, #2ecc71, #27ae60);">
                                    <div class="metric-value" id="highPrice">₹0.00</div>
                                    <div class="metric-label">52W High</div>
                                </div>
                                <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                                    <div class="metric-value" id="lowPrice">₹0.00</div>
                                    <div class="metric-label">52W Low</div>
                                </div>
                                <div class="metric-card" style="background: linear-gradient(135deg, #f39c12, #e67e22);">
                                    <div class="metric-value" id="volatility">0.0%</div>
                                    <div class="metric-label">Volatility</div>
                                </div>
                            </div>

                            <div class="chart-container">
                                <canvas id="priceChart" width="800" height="400"></canvas>
                            </div>
                        </div>

                        <!-- Model Performance -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <i class="fas fa-trophy"></i>
                                <h3>Model Performance</h3>
                            </div>
                            
                            <div class="feature-grid" id="modelResults">
                                <!-- Model results will be populated here -->
                            </div>
                        </div>

                        <!-- Predictions Visualization -->
                        <div class="analysis-section">
                            <div class="section-header">
                                <i class="fas fa-crystal-ball"></i>
                                <h3>Future Price Predictions</h3>
                            </div>
                            
                            <div class="chart-container">
                                <canvas id="predictionChart" width="800" height="400"></canvas>
                            </div>

                            <!-- Investment Recommendation -->
                            <div class="alert alert-success" id="recommendation" style="display: none;">
                                <h3>Investment Recommendation</h3>
                                <div style="text-align: center; margin: 1rem 0;">
                                    <div style="font-size: 2rem; font-weight: bold;" id="recommendationText">ANALYZING...</div>
                                    <div style="margin-top: 0.5rem;" id="recommendationDetails">Please wait while we analyze the predictions...</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Portfolio Tab -->
                <div id="portfolioTab" class="tab-content">
                    <div class="analysis-section fade-in">
                        <div class="section-header">
                            <i class="fas fa-briefcase"></i>
                            <h3>Portfolio Analysis</h3>
                        </div>
                        
                        <div class="alert alert-info">
                            <h4>🎯 Portfolio Management Features</h4>
                            <p>Advanced portfolio analysis tools including:</p>
                            <ul style="margin: 1rem 0; padding-left: 1.5rem;">
                                <li>Multi-stock correlation analysis</li>
                                <li>Risk-return optimization</li>
                                <li>Diversification recommendations</li>
                                <li>Performance tracking</li>
                                <li>Rebalancing suggestions</li>
                            </ul>
                        </div>

                        <div class="feature-grid">
                            <div class="feature-card">
                                <span class="feature-icon">📊</span>
                                <h3 class="feature-title">Correlation Matrix</h3>
                                <p class="feature-description">Analyze how your stocks move together and identify diversification opportunities.</p>
                                <button class="btn btn-primary" style="margin-top: 1rem;">
                                    <i class="fas fa-play"></i>
                                    Coming Soon
                                </button>
                            </div>
                            <div class="feature-card">
                                <span class="feature-icon">⚖️</span>
                                <h3 class="feature-title">Risk Assessment</h3>
                                <p class="feature-description">Evaluate portfolio risk using VaR, Sharpe ratio, and beta calculations.</p>
                                <button class="btn btn-primary" style="margin-top: 1rem;">
                                    <i class="fas fa-play"></i>
                                    Coming Soon
                                </button>
                            </div>
                            <div class="feature-card">
                                <span class="feature-icon">🎯</span>
                                <h3 class="feature-title">Optimization</h3>
                                <p class="feature-description">Modern portfolio theory optimization for maximum risk-adjusted returns.</p>
                                <button class="btn btn-primary" style="margin-top: 1rem;">
                                    <i class="fas fa-play"></i>
                                    Coming Soon
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Real-Time Tab -->
                <div id="realtimeTab" class="tab-content">
                    <div class="analysis-section fade-in">
                        <div class="section-header">
                            <i class="fas fa-bolt"></i>
                            <h3>Real-Time Market Monitoring</h3>
                        </div>
                        
                        <div class="alert alert-warning">
                            <h4>⚡ Live Market Features</h4>
                            <p>Real-time market monitoring capabilities:</p>
                            <ul style="margin: 1rem 0; padding-left: 1.5rem;">
                                <li>Live price feeds and updates</li>
                                <li>Custom price alerts and notifications</li>
                                <li>Volume and momentum indicators</li>
                                <li>Breaking news integration</li>
                                <li>Market sentiment analysis</li>
                            </ul>
                        </div>

                        <div class="metrics-grid">
                            <div class="metric-card pulse">
                                <div class="metric-value">LIVE</div>
                                <div class="metric-label">Market Status</div>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                                <div class="metric-value">15:30</div>
                                <div class="metric-label">Market Close</div>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, #f39c12, #e67e22);">
                                <div class="metric-value">45,287</div>
                                <div class="metric-label">Nifty 50</div>
                            </div>
                            <div class="metric-card" style="background: linear-gradient(135deg, #9b59b6, #8e44ad);">
                                <div class="metric-value">67,543</div>
                                <div class="metric-label">Sensex</div>
                            </div>
                        </div>

                        <div class="chart-container">
                            <h4 style="margin-bottom: 1rem;">Live Price Simulation</h4>
                            <canvas id="liveChart" width="800" height="300"></canvas>
                        </div>
                    </div>
                </div>

                <!-- AI Insights Tab -->
                <div id="insightsTab" class="tab-content">
                    <div class="analysis-section fade-in">
                        <div class="section-header">
                            <i class="fas fa-brain"></i>
                            <h3>AI Market Insights</h3>
                        </div>
                        
                        <button class="btn btn-primary" onclick="generateInsights()" style="margin-bottom: 2rem;">
                            <i class="fas fa-magic"></i>
                            Generate AI Insights
                        </button>

                        <div id="insightsContainer">
                            <div class="alert alert-info">
                                <h4>🤖 AI-Powered Market Analysis</h4>
                                <p>Our AI system analyzes multiple data sources to provide actionable insights:</p>
                                <ul style="margin: 1rem 0; padding-left: 1.5rem;">
                                    <li>Technical pattern recognition</li>
                                    <li>Market sentiment analysis</li>
                                    <li>News impact assessment</li>
                                    <li>Sector rotation predictions</li>
                                    <li>Risk-adjusted recommendations</li>
                                </ul>
                                <p>Click "Generate AI Insights" to get personalized market analysis.</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Market Scanner Tab -->
                <div id="scannerTab" class="tab-content">
                    <div class="analysis-section fade-in">
                        <div class="section-header">
                            <i class="fas fa-search"></i>
                            <h3>Market Scanner</h3>
                        </div>
                        
                        <div class="form-group" style="max-width: 400px; margin-bottom: 2rem;">
                            <label class="form-label" style="color: var(--text-primary);">Scan Criteria</label>
                            <select class="form-select" id="scanCriteria" style="background: var(--bg-secondary); color: var(--text-primary); border: 1px solid var(--border-color);">
                                <option value="rsi_oversold">RSI Oversold (< 30)</option>
                                <option value="rsi_overbought">RSI Overbought (> 70)</option>
                                <option value="volume_breakout">High Volume Breakouts</option>
                                <option value="near_52w_high">Price near 52W High</option>
                                <option value="near_52w_low">Price near 52W Low</option>
                                <option value="bullish_pattern">Bullish Chart Patterns</option>
                                <option value="bearish_pattern">Bearish Chart Patterns</option>
                            </select>
                        </div>

                        <button class="btn btn-primary" onclick="scanMarket()" style="margin-bottom: 2rem;">
                            <i class="fas fa-search"></i>
                            Scan Market
                        </button>

                        <div id="scanResults">
                            <div class="alert alert-info">
                                <h4>📊 Market Screening Tool</h4>
                                <p>Advanced market scanner to identify trading opportunities:</p>
                                <ul style="margin: 1rem 0; padding-left: 1.5rem;">
                                    <li>Technical indicator screening</li>
                                    <li>Volume and momentum analysis</li>
                                    <li>Pattern recognition algorithms</li>
                                    <li>Custom filter combinations</li>
                                    <li>Real-time opportunity alerts</li>
                                </ul>
                                <p>Select your criteria and click "Scan Market" to find opportunities.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        // Theme Management
        let currentTheme = 'light';
        
        function toggleTheme() {
            currentTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.body.setAttribute('data-theme', currentTheme);
            
            const themeIcon = document.getElementById('themeIcon');
            const themeText = document.getElementById('themeText');
            
            if (currentTheme === 'dark') {
                themeIcon.className = 'fas fa-sun';
                themeText.textContent = 'Light Mode';
            } else {
                themeIcon.className = 'fas fa-moon';
                themeText.textContent = 'Dark Mode';
            }
            
            // Update charts if they exist
            updateChartThemes();
        }

        // Sidebar Management
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebar.classList.toggle('collapsed');
        }

        // Stock Filtering
        const stocks = {
            'TCS.NS': 'Tata Consultancy Services',
            'INFY.NS': 'Infosys Limited',
            'RELIANCE.NS': 'Reliance Industries',
            'HDFCBANK.NS': 'HDFC Bank',
            'ICICIBANK.NS': 'ICICI Bank',
            'SBIN.NS': 'State Bank of India',
            'WIPRO.NS': 'Wipro Limited',
            'TECHM.NS': 'Tech Mahindra',
            'MARUTI.NS': 'Maruti Suzuki',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'ITC.NS': 'ITC Limited',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'AXISBANK.NS': 'Axis Bank Limited',
            'BAJFINANCE.NS': 'Bajaj Finance',
            'LT.NS': 'Larsen & Toubro',
            'SUNPHARMA.NS': 'Sun Pharmaceutical',
            'TATASTEEL.NS': 'Tata Steel',
            'ONGC.NS': 'Oil and Natural Gas Corporation',
            'COALINDIA.NS': 'Coal India Limited',
            'TITAN.NS': 'Titan Company Limited'
        };

        function filterStocks() {
            const searchTerm = document.getElementById('stockSearch').value.toLowerCase();
            const stockSelect = document.getElementById('stockSelect');
            
            // Clear current options
            stockSelect.innerHTML = '';
            
            // Filter and add matching stocks
            Object.entries(stocks).forEach(([symbol, name]) => {
                if (symbol.toLowerCase().includes(searchTerm) || name.toLowerCase().includes(searchTerm)) {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = `${symbol} - ${name}`;
                    stockSelect.appendChild(option);
                }
            });
        }

        // Slider Updates
        function updateSliderValue(sliderId, valueId) {
            const slider = document.getElementById(sliderId);
            const valueDisplay = document.getElementById(valueId);
            valueDisplay.textContent = slider.value;
        }

        // Tab Management
        function switchTab(tabName) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName + 'Tab').classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }

        // Analysis Functions
        let analysisInProgress = false;

        function startAnalysis() {
            if (analysisInProgress) return;
            
            // Validate model selection
            const models = ['linearModel', 'lstmModel', 'prophetModel'];
            const selectedModels = models.filter(model => document.getElementById(model).checked);
            
            if (selectedModels.length === 0) {
                alert('Please select at least one model!');
                return;
            }

            analysisInProgress = true;
            
            // Hide welcome screen and show results
            document.getElementById('welcomeScreen').style.display = 'none';
            document.getElementById('analysisResults').style.display = 'block';
            
            // Start analysis simulation
            simulateAnalysis();
        }

        function simulateAnalysis() {
            const steps = [
                { message: '🚀 Initializing AI system...', progress: 10 },
                { message: '📡 Fetching market data...', progress: 25 },
                { message: '🔧 Processing data and indicators...', progress: 40 },
                { message: '🤖 Training Linear Regression...', progress: 55 },
                { message: '🧠 Training LSTM Neural Network...', progress: 70 },
                { message: '📊 Training Prophet model...', progress: 85 },
                { message: '📈 Generating predictions...', progress: 95 },
                { message: '✅ Analysis complete!', progress: 100 }
            ];

            let currentStep = 0;
            
            const updateProgress = () => {
                if (currentStep < steps.length) {
                    const step = steps[currentStep];
                    updateStatus(step.message, step.progress);
                    currentStep++;
                    setTimeout(updateProgress, 1500);
                } else {
                    completeAnalysis();
                }
            };

            updateProgress();
        }

        function updateStatus(message, progress) {
            const statusMessage = document.getElementById('statusMessage');
            const progressFill = document.getElementById('progressFill');
            
            statusMessage.innerHTML = `<i class="fas fa-spinner fa-spin"></i><span>${message}</span>`;
            progressFill.style.width = progress + '%';
        }

        function completeAnalysis() {
            const statusMessage = document.getElementById('statusMessage');
            statusMessage.innerHTML = '<i class="fas fa-check-circle"></i><span>Analysis complete! Results are ready.</span>';
            statusMessage.className = 'status-message status-success';
            
            // Populate mock results
            populateStockData();
            populateModelResults();
            createPredictionChart();
            showRecommendation();
            
            analysisInProgress = false;
        }

        function populateStockData() {
            // Mock stock data
            const currentPrice = 3245.50;
            const prevPrice = 3180.25;
            const change = currentPrice - prevPrice;
            const changePct = (change / prevPrice) * 100;
            
            document.getElementById('currentPrice').textContent = `₹${currentPrice.toFixed(2)}`;
            document.getElementById('priceChange').textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)} (${changePct.toFixed(2)}%)`;
            document.getElementById('priceChange').style.color = change > 0 ? '#2ecc71' : '#e74c3c';
            
            document.getElementById('highPrice').textContent = '₹3,456.80';
            document.getElementById('lowPrice').textContent = '₹2,890.45';
            document.getElementById('volatility').textContent = '28.5%';
            
            // Create price chart
            createPriceChart();
        }

        function populateModelResults() {
            const models = [
                {
                    name: 'Linear Regression',
                    icon: '📊',
                    rmse: '45.23',
                    r2: '0.8634',
                    mape: '2.1',
                    accuracy: '78.5',
                    isBest: false
                },
                {
                    name: 'LSTM Neural Network',
                    icon: '🧠',
                    rmse: '38.67',
                    r2: '0.9145',
                    mape: '1.8',
                    accuracy: '85.2',
                    isBest: true
                },
                {
                    name: 'Prophet',
                    icon: '📈',
                    rmse: '42.15',
                    r2: '0.8823',
                    mape: '1.9',
                    accuracy: '81.7',
                    isBest: false
                }
            ];

            const container = document.getElementById('modelResults');
            container.innerHTML = '';

            models.forEach(model => {
                const card = document.createElement('div');
                card.className = 'feature-card';
                if (model.isBest) {
                    card.style.border = '2px solid var(--success-color)';
                    card.style.boxShadow = '0 0 20px rgba(79, 172, 254, 0.3)';
                }

                card.innerHTML = `
                    <span class="feature-icon">${model.isBest ? '👑' : ''}${model.icon}</span>
                    <h3 class="feature-title">${model.name}</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 1rem;">
                        <div><strong>RMSE:</strong> ₹${model.rmse}</div>
                        <div><strong>R²:</strong> ${model.r2}</div>
                        <div><strong>MAPE:</strong> ${model.mape}%</div>
                        <div><strong>Accuracy:</strong> ${model.accuracy}%</div>
                    </div>
                    ${model.isBest ? '<div style="color: var(--success-color); margin-top: 0.5rem; font-weight: bold;">🏆 Best Performer</div>' : ''}
                `;
                
                container.appendChild(card);
            });
        }

        function createPriceChart() {
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            // Generate mock price data
            const dates = [];
            const prices = [];
            const basePrice = 3000;
            
            for (let i = 30; i >= 0; i--) {
                const date = new Date();
                date.setDate(date.getDate() - i);
                dates.push(date.toLocaleDateString());
                
                const randomChange = (Math.random() - 0.5) * 100;
                const price = basePrice + randomChange + (Math.sin(i / 5) * 50);
                prices.push(price.toFixed(2));
            }

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [{
                        label: 'Price (₹)',
                        data: prices,
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Stock Price History'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }

        function createPredictionChart() {
            const ctx = document.getElementById('predictionChart').getContext('2d');
            
            // Generate mock prediction data
            const dates = [];
            const currentPrice = 3245.50;
            const linearPred = [];
            const lstmPred = [];
            const prophetPred = [];
            
            for (let i = 1; i <= 30; i++) {
                const date = new Date();
                date.setDate(date.getDate() + i);
                dates.push(date.toLocaleDateString());
                
                // Mock predictions with different trends
                linearPred.push((currentPrice + (i * 2.5) + (Math.random() - 0.5) * 20).toFixed(2));
                lstmPred.push((currentPrice + (i * 3.2) + Math.sin(i / 3) * 15 + (Math.random() - 0.5) * 15).toFixed(2));
                prophetPred.push((currentPrice + (i * 2.8) + Math.cos(i / 4) * 25 + (Math.random() - 0.5) * 18).toFixed(2));
            }

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'Linear Regression',
                            data: linearPred,
                            borderColor: 'rgb(31, 119, 180)',
                            backgroundColor: 'rgba(31, 119, 180, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'LSTM Neural Network',
                            data: lstmPred,
                            borderColor: 'rgb(255, 127, 14)',
                            backgroundColor: 'rgba(255, 127, 14, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Prophet',
                            data: prophetPred,
                            borderColor: 'rgb(44, 160, 44)',
                            backgroundColor: 'rgba(44, 160, 44, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Future Price Predictions (30 Days)'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }

        function showRecommendation() {
            const recommendation = document.getElementById('recommendation');
            const recommendationText = document.getElementById('recommendationText');
            const recommendationDetails = document.getElementById('recommendationDetails');
            
            // Mock recommendation logic
            const avgChange = 5.8; // Simulated average predicted change
            
            let signal, color, details;
            
            if (avgChange > 5) {
                signal = '🟢 STRONG BUY';
                color = '#2ecc71';
                details = `Average predicted increase: +${avgChange.toFixed(1)}% over 30 days. Strong bullish signals across all models.`;
                recommendation.className = 'alert alert-success';
            } else if (avgChange > 2) {
                signal = '🟢 BUY';
                color = '#2ecc71';
                details = `Average predicted increase: +${avgChange.toFixed(1)}% over 30 days. Moderate bullish signals.`;
                recommendation.className = 'alert alert-success';
            } else if (avgChange > -2) {
                signal = '🟡 HOLD';
                color = '#f39c12';
                details = `Average predicted change: ${avgChange.toFixed(1)}% over 30 days. Mixed signals, consider holding.`;
                recommendation.className = 'alert alert-warning';
            } else {
                signal = '🔴 SELL';
                color = '#e74c3c';
                details = `Average predicted decrease: ${avgChange.toFixed(1)}% over 30 days. Bearish signals detected.`;
                recommendation.className = 'alert alert-warning';
            }
            
            recommendationText.textContent = signal;
            recommendationText.style.color = color;
            recommendationDetails.textContent = details;
            recommendation.style.display = 'block';
        }

        function clearCache() {
            // Reset to welcome screen
            document.getElementById('welcomeScreen').style.display = 'block';
            document.getElementById('analysisResults').style.display = 'none';
            
            // Reset progress
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('statusMessage').innerHTML = '<i class="fas fa-info-circle"></i><span>Ready to start analysis...</span>';
            document.getElementById('statusMessage').className = 'status-message status-info';
            
            // Reset forms
            document.getElementById('stockSearch').value = '';
            filterStocks();
            
            alert('Cache cleared successfully!');
        }

        function refreshData() {
            document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
            
            // Add refresh animation
            const refreshBtn = event.target;
            refreshBtn.style.transform = 'rotate(360deg)';
            setTimeout(() => {
                refreshBtn.style.transform = 'rotate(0deg)';
            }, 500);
        }

        // AI Insights Functions
        function generateInsights() {
            const container = document.getElementById('insightsContainer');
            container.innerHTML = '<div class="status-message status-info"><i class="fas fa-spinner fa-spin"></i><span>Generating AI insights...</span></div>';
            
            setTimeout(() => {
                const insights = [
                    {
                        icon: '💡',
                        title: 'Market Volatility Alert',
                        content: 'Market volatility has increased 15% over the past week due to global economic uncertainties. Consider defensive positions.',
                        type: 'warning'
                    },
                    {
                        icon: '📊',
                        title: 'Banking Sector Momentum',
                        content: 'Banking sector shows strong technical breakouts with HDFC Bank and ICICI Bank leading the charge. RSI levels suggest continuation.',
                        type: 'success'
                    },
                    {
                        icon: '⚠️',
                        title: 'IT Sector Headwinds',
                        content: 'IT stocks may face near-term headwinds due to US recession fears and currency fluctuations. Monitor closely.',
                        type: 'warning'
                    },
                    {
                        icon: '🎯',
                        title: 'Energy Opportunity',
                        content: 'Energy sector presents buying opportunities on recent dips. Oil prices stabilizing and demand recovery expected.',
                        type: 'success'
                    },
                    {
                        icon: '📈',
                        title: 'Defensive Allocation',
                        content: 'Consider increasing allocation to defensive sectors like FMCG and Healthcare given current market uncertainty.',
                        type: 'info'
                    }
                ];

                let html = '';
                insights.forEach(insight => {
                    html += `
                        <div class="alert alert-${insight.type} fade-in" style="margin: 1rem 0;">
                            <h4>${insight.icon} ${insight.title}</h4>
                            <p style="margin: 0.5rem 0 0 0;">${insight.content}</p>
                        </div>
                    `;
                });

                container.innerHTML = html;
            }, 2000);
        }

        // Market Scanner Functions
        function scanMarket() {
            const criteria = document.getElementById('scanCriteria').value;
            const container = document.getElementById('scanResults');
            
            container.innerHTML = '<div class="status-message status-info"><i class="fas fa-spinner fa-spin"></i><span>Scanning market...</span></div>';
            
            setTimeout(() => {
                const mockResults = generateMockScanResults(criteria);
                displayScanResults(mockResults, criteria);
            }, 1500);
        }

        function generateMockScanResults(criteria) {
            const results = {
                'rsi_oversold': [
                    { symbol: 'TATASTEEL.NS', name: 'Tata Steel', price: '₹1,245.30', change: '-2.8%', rsi: '28.5' },
                    { symbol: 'COALINDIA.NS', name: 'Coal India', price: '₹189.45', change: '-1.2%', rsi: '29.1' },
                    { symbol: 'ONGC.NS', name: 'ONGC', price: '₹156.78', change: '-0.9%', rsi: '27.8' }
                ],
                'rsi_overbought': [
                    { symbol: 'BAJFINANCE.NS', name: 'Bajaj Finance', price: '₹7,856.90', change: '+3.2%', rsi: '72.4' },
                    { symbol: 'TITAN.NS', name: 'Titan Company', price: '₹3,234.56', change: '+2.1%', rsi: '71.8' }
                ],
                'volume_breakout': [
                    { symbol: 'RELIANCE.NS', name: 'Reliance Industries', price: '₹2,789.45', change: '+4.5%', volume: '250%' },
                    { symbol: 'TCS.NS', name: 'TCS', price: '₹3,245.50', change: '+2.8%', volume: '180%' },
                    { symbol: 'INFY.NS', name: 'Infosys', price: '₹1,456.30', change: '+1.9%', volume: '165%' }
                ],
                'near_52w_high': [
                    { symbol: 'HDFCBANK.NS', name: 'HDFC Bank', price: '₹1,687.90', change: '+1.5%', distance: '2.1%' },
                    { symbol: 'ICICIBANK.NS', name: 'ICICI Bank', price: '₹945.67', change: '+0.8%', distance: '1.8%' }
                ],
                'near_52w_low': [
                    { symbol: 'WIPRO.NS', name: 'Wipro', price: '₹398.45', change: '-1.2%', distance: '3.2%' },
                    { symbol: 'TECHM.NS', name: 'Tech Mahindra', price: '₹1,089.34', change: '-0.5%', distance: '2.9%' }
                ],
                'bullish_pattern': [
                    { symbol: 'MARUTI.NS', name: 'Maruti Suzuki', price: '₹9,876.54', change: '+2.3%', pattern: 'Cup & Handle' },
                    { symbol: 'LT.NS', name: 'L&T', price: '₹2,456.78', change: '+1.7%', pattern: 'Ascending Triangle' }
                ],
                'bearish_pattern': [
                    { symbol: 'ITC.NS', name: 'ITC', price: '₹456.78', change: '-1.8%', pattern: 'Head & Shoulders' },
                    { symbol: 'SUNPHARMA.NS', name: 'Sun Pharma', price: '₹1,123.45', change: '-0.9%', pattern: 'Double Top' }
                ]
            };

            return results[criteria] || [];
        }

        function displayScanResults(results, criteria) {
            const container = document.getElementById('scanResults');
            
            if (results.length === 0) {
                container.innerHTML = `
                    <div class="alert alert-warning">
                        <h4>📊 No Results Found</h4>
                        <p>No stocks currently match the selected criteria. Try different scan parameters.</p>
                    </div>
                `;
                return;
            }

            const criteriaNames = {
                'rsi_oversold': 'RSI Oversold Stocks',
                'rsi_overbought': 'RSI Overbought Stocks',
                'volume_breakout': 'High Volume Breakouts',
                'near_52w_high': 'Near 52-Week High',
                'near_52w_low': 'Near 52-Week Low',
                'bullish_pattern': 'Bullish Chart Patterns',
                'bearish_pattern': 'Bearish Chart Patterns'
            };

            let html = `
                <div class="alert alert-success">
                    <h4>🎯 Found ${results.length} opportunities for ${criteriaNames[criteria]}</h4>
                </div>
                <div class="analysis-section">
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                            <thead>
                                <tr style="background: var(--bg-secondary); border-bottom: 2px solid var(--border-color);">
                                    <th style="padding: 1rem; text-align: left; color: var(--text-primary);">Symbol</th>
                                    <th style="padding: 1rem; text-align: left; color: var(--text-primary);">Company</th>
                                    <th style="padding: 1rem; text-align: right; color: var(--text-primary);">Price</th>
                                    <th style="padding: 1rem; text-align: right; color: var(--text-primary);">Change</th>
                                    <th style="padding: 1rem; text-align: right; color: var(--text-primary);">Signal</th>
                                </tr>
                            </thead>
                            <tbody>
            `;

            results.forEach((stock, index) => {
                const signalValue = stock.rsi || stock.volume || stock.distance || stock.pattern || 'N/A';
                const changeColor = stock.change.startsWith('+') ? '#2ecc71' : '#e74c3c';
                
                html += `
                    <tr style="border-bottom: 1px solid var(--border-color); transition: all 0.3s ease;" 
                        onmouseover="this.style.background='var(--bg-secondary)'" 
                        onmouseout="this.style.background='transparent'">
                        <td style="padding: 1rem; font-weight: bold; color: var(--primary-color);">${stock.symbol}</td>
                        <td style="padding: 1rem; color: var(--text-primary);">${stock.name}</td>
                        <td style="padding: 1rem; text-align: right; color: var(--text-primary);">${stock.price}</td>
                        <td style="padding: 1rem; text-align: right; color: ${changeColor}; font-weight: bold;">${stock.change}</td>
                        <td style="padding: 1rem; text-align: right; color: var(--text-secondary);">${signalValue}</td>
                    </tr>
                `;
            });

            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;

            container.innerHTML = html;
        }

        // Live Chart for Real-time Tab
        let liveChart;
        
        function createLiveChart() {
            const ctx = document.getElementById('liveChart');
            if (!ctx) return;
            
            const chartCtx = ctx.getContext('2d');
            
            // Generate initial data
            const data = [];
            const labels = [];
            const basePrice = 3245.50;
            
            for (let i = 0; i < 50; i++) {
                const time = new Date(Date.now() - (50 - i) * 1000);
                labels.push(time.toLocaleTimeString());
                data.push(basePrice + (Math.random() - 0.5) * 20);
            }

            liveChart = new Chart(chartCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Live Price',
                        data: data,
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    animation: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Live Price Feed Simulation'
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });

            // Simulate live updates
            setInterval(() => {
                if (liveChart && document.getElementById('realtimeTab').classList.contains('active')) {
                    const newTime = new Date().toLocaleTimeString();
                    const newPrice = basePrice + (Math.random() - 0.5) * 20;
                    
                    liveChart.data.labels.push(newTime);
                    liveChart.data.datasets[0].data.push(newPrice);
                    
                    // Keep only last 50 points
                    if (liveChart.data.labels.length > 50) {
                        liveChart.data.labels.shift();
                        liveChart.data.datasets[0].data.shift();
                    }
                    
                    liveChart.update('none');
                }
            }, 1000);
        }

        // Chart theme updates
        function updateChartThemes() {
            // This would update existing charts when theme changes
            // Implementation depends on having chart instances available
        }

        // Mobile responsiveness
        function handleMobileMenu() {
            if (window.innerWidth <= 768) {
                const sidebar = document.getElementById('sidebar');
                sidebar.classList.add('mobile');
            }
        }

        // Initialize app
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize stock list
            filterStocks();
            
            // Set initial last updated time
            document.getElementById('lastUpdated').textContent = 'Never';
            
            // Create live chart after a delay to ensure tab is loaded
            setTimeout(() => {
                createLiveChart();
            }, 1000);
            
            // Handle mobile responsiveness
            handleMobileMenu();
            window.addEventListener('resize', handleMobileMenu);
            
            // Add smooth scrolling
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });

            // Add keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                // Ctrl/Cmd + K for search
                if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                    e.preventDefault();
                    document.getElementById('stockSearch').focus();
                }
                
                // Ctrl/Cmd + Enter for analysis
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    e.preventDefault();
                    startAnalysis();
                }
                
                // Ctrl/Cmd + D for theme toggle
                if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
                    e.preventDefault();
                    toggleTheme();
                }
            });

            // Add tooltips for better UX
            const tooltips = document.querySelectorAll('[title]');
            tooltips.forEach(element => {
                element.addEventListener('mouseenter', function() {
                    const tooltip = document.createElement('div');
                    tooltip.className = 'tooltip';
                    tooltip.textContent = this.getAttribute('title');
                    tooltip.style.cssText = `
                        position: absolute;
                        background: var(--bg-tertiary);
                        color: var(--text-primary);
                        padding: 0.5rem;
                        border-radius: 4px;
                        font-size: 0.875rem;
                        z-index: 1000;
                        pointer-events: none;
                        box-shadow: var(--shadow-medium);
                        border: 1px solid var(--border-color);
                    `;
                    document.body.appendChild(tooltip);
                    
                    const rect = this.getBoundingClientRect();
                    tooltip.style.left = rect.left + 'px';
                    tooltip.style.top = (rect.bottom + 5) + 'px';
                    
                    this.tooltip = tooltip;
                });
                
                element.addEventListener('mouseleave', function() {
                    if (this.tooltip) {
                        document.body.removeChild(this.tooltip);
                        this.tooltip = null;
                    }
                });
            });

            // Animate elements on scroll
            const observerOptions = {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            };

            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            }, observerOptions);

            // Observe all cards and sections
            document.querySelectorAll('.feature-card, .analysis-section, .metric-card').forEach(el => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';
                el.style.transition = 'all 0.6s ease';
                observer.observe(el);
            });

            // Performance monitoring
            if ('PerformanceObserver' in window) {
                const perfObserver = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        if (entry.entryType === 'measure') {
                            console.log(`${entry.name}: ${entry.duration}ms`);
                        }
                    }
                });
                perfObserver.observe({entryTypes: ['measure']});
            }

            console.log('🚀 Enhanced Stock Prediction App Initialized');
            console.log('💡 Keyboard shortcuts:');
            console.log('   Ctrl/Cmd + K: Focus search');
            console.log('   Ctrl/Cmd + Enter: Start analysis');
            console.log('   Ctrl/Cmd + D: Toggle theme');
        });
    </script>
</body>
</html>