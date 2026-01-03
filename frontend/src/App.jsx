import { useState, useEffect } from 'react';
import { 
  Wine, 
  Sparkles, 
  TrendingUp, 
  Info, 
  BarChart3, 
  Award,
  Thermometer,
  Zap,
  Droplets,
  Scale,
  Beaker,
  Shield,
  Percent,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';
import './App.css';

// ========================================
// API CONFIGURATION
// ========================================
const API_URL = "https://ml-lab-model-2.onrender.com";

// ========================================
// COMPONENT: WineQualityPredictor
// ========================================
function App() {
  // ========================================
  // STATE MANAGEMENT
  // ========================================
  const [formData, setFormData] = useState({
    fixed_acidity: 7.4,
    volatile_acidity: 0.7,
    citric_acid: 0.0,
    residual_sugar: 1.9,
    chlorides: 0.076,
    free_sulfur_dioxide: 11.0,
    total_sulfur_dioxide: 34.0,
    density: 0.9978,
    pH: 3.51,
    sulphates: 0.56,
    alcohol: 9.4
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [options, setOptions] = useState(null);

  // ========================================
  // EFFECTS
  // ========================================
  
  // Fetch typical ranges on component mount
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch(`${API_URL}/api/options`);
        if (!response.ok) throw new Error('Failed to fetch options');
        const data = await response.json();
        setOptions(data);
      } catch (err) {
        console.error('Failed to load options:', err);
        setError('Unable to load wine property ranges');
      }
    };

    fetchOptions();
  }, []);

  // ========================================
  // EVENT HANDLERS
  // ========================================
  
  /**
   * Handle form input changes
   */
  const handleChange = (name, value) => {
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      setFormData(prev => ({
        ...prev,
        [name]: parsedValue
      }));
    }
  };

  /**
   * Load preset wine profiles
   */
  const loadPreset = (preset) => {
    const presets = {
      Substandard: {
        fixed_acidity: 7.8,
        volatile_acidity: 1.2,
        citric_acid: 0.0,
        residual_sugar: 2.0,
        chlorides: 0.12,
        free_sulfur_dioxide: 6.0,
        total_sulfur_dioxide: 15.0,
        density: 0.9968,
        pH: 3.7,
        sulphates: 0.4,
        alcohol: 8.5
      },
      Satisfactory: {
        fixed_acidity: 7.4,
        volatile_acidity: 0.7,
        citric_acid: 0.0,
        residual_sugar: 1.9,
        chlorides: 0.076,
        free_sulfur_dioxide: 11.0,
        total_sulfur_dioxide: 34.0,
        density: 0.9978,
        pH: 3.51,
        sulphates: 0.56,
        alcohol: 9.4
      },
      Premium: {
        fixed_acidity: 7.5,
        volatile_acidity: 0.25,
        citric_acid: 0.45,
        residual_sugar: 2.0,
        chlorides: 0.09,
        free_sulfur_dioxide: 22.0,
        total_sulfur_dioxide: 65.0,
        density: 0.992,
        pH: 3.25,
        sulphates: 1.05,
        alcohol: 13.8
      }
    };
    
    setFormData(presets[preset]);
    setResult(null);
    setError(null);
  };

  /**
   * Handle form submission
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    // Don't clear result here - only clear on successful new prediction
    // setResult(null); // REMOVED: This was causing the issue

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  // ========================================
  // HELPER FUNCTIONS
  // ========================================
  
  /**
   * Get configuration for a specific feature
   */
  const getFeatureConfig = (feature) => {
    if (!options) return { min: 0, max: 100, step: 0.1 };
    
    const ranges = options.typical_ranges?.[feature];
    if (!ranges) return { min: 0, max: 100, step: 0.1 };

    return {
      min: ranges.min,
      max: ranges.max,
      step: feature === 'pH' || feature.includes('acidity') ? 0.01 : 
            feature.includes('sulfur') || feature === 'alcohol' ? 0.1 : 0.001
    };
  };

  /**
   * Format feature names for display
   */
  const QUALITY_ORDER = [ "Premium", "Satisfactory", "Substandard" ];
  const formatFeatureName = (feature) => {
    return feature
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  /**
   * Get icon for a feature category
   */
  const getFeatureIcon = (feature) => {
    const icons = {
      fixed_acidity: <Thermometer size={16} />,
      volatile_acidity: <Zap size={16} />,
      citric_acid: <Droplets size={16} />,
      pH: <Scale size={16} />,
      residual_sugar: <Beaker size={16} />,
      chlorides: <Shield size={16} />,
      free_sulfur_dioxide: <Sparkles size={16} />,
      total_sulfur_dioxide: <Sparkles size={16} />,
      sulphates: <Percent size={16} />,
      alcohol: <Wine size={16} />,
      density: <BarChart3 size={16} />
    };
    
    return icons[feature] || <Info size={16} />;
  };

  // ========================================
  // LOADING STATE
  // ========================================
  if (!options) {
    return (
      <div className="app">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading wine analyzer...</p>
        </div>
      </div>
    );
  }

  // ========================================
  // RENDER COMPONENT
  // ========================================
  return (
    <div className="app">
      <div className="app-container">
        {/* ========================================
            HEADER SECTION
            ======================================== */}
        <header className="header">
          <div className="header-content">
            <div className="logo-container">
              <Wine size={36} className="logo" />
            </div>
            <div className="header-text">
              <h1>
                Wine Quality Predictor
                <span className="model-badge">Decision Tree ML</span>
              </h1>
              <p>
                AI-powered wine quality assessment from physicochemical analysis
              </p>
            </div>
          </div>
        </header>

        {/* ========================================
            MAIN CONTENT SECTION
            ======================================== */}
        <main className="main">
          <div className="main-grid">
            
            {/* ========================================
                INPUT FORM CARD
                ======================================== */}
            <div className="card form-card">
              <div className="card-header">
                <Sparkles size={24} />
                <h2>Wine Properties</h2>
              </div>

              <div className="form">
                <div className="form-section">
                  <h3 className="section-title">
                    <TrendingUp size={20} />
                    <span>How It Works</span>
                  </h3>
                  <p style={{ color: '#a1a1a1', lineHeight: 1.6, fontSize: '0.95rem' }}>
                    Enter the physicochemical properties of your wine sample. 
                    Our Decision Tree model analyzes 11 key features to predict quality.
                  </p>
                </div>

                <form onSubmit={handleSubmit}>
                  
                  {/* Acidity & pH Section */}
                  <div className="form-section">
                    <h3 className="section-title">
                      <Thermometer size={20} />
                      <span>Acidity & pH Balance</span>
                    </h3>
                    <div className="form-grid">
                      {['fixed_acidity', 'volatile_acidity', 'citric_acid', 'pH'].map(feature => {
                        const config = getFeatureConfig(feature);
                        const value = formData[feature];
                        
                        return (
                          <div key={feature} className="form-group">
                            <label>
                              <span>
                                {getFeatureIcon(feature)}
                                {formatFeatureName(feature)}
                              </span>
                              <span className="label-value">
                                {value.toFixed(config.step === 0.001 ? 3 : 2)}
                              </span>
                            </label>
                            <input
                              type="range"
                              min={config.min}
                              max={config.max}
                              step={config.step}
                              value={value}
                              onChange={(e) => handleChange(feature, e.target.value)}
                              aria-label={formatFeatureName(feature)}
                            />
                            <div className="range-labels">
                              <span>{config.min.toFixed(1)}</span>
                              <span>{config.max.toFixed(1)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Composition Section */}
                  <div className="form-section">
                    <h3 className="section-title">
                      <BarChart3 size={20} />
                      <span>Chemical Composition</span>
                    </h3>
                    <div className="form-grid">
                      {['residual_sugar', 'chlorides', 'density', 'sulphates'].map(feature => {
                        const config = getFeatureConfig(feature);
                        const value = formData[feature];
                        
                        return (
                          <div key={feature} className="form-group">
                            <label>
                              <span>
                                {getFeatureIcon(feature)}
                                {formatFeatureName(feature)}
                              </span>
                              <span className="label-value">
                                {value.toFixed(config.step === 0.001 ? 3 : 2)}
                              </span>
                            </label>
                            <input
                              type="range"
                              min={config.min}
                              max={config.max}
                              step={config.step}
                              value={value}
                              onChange={(e) => handleChange(feature, e.target.value)}
                              aria-label={formatFeatureName(feature)}
                            />
                            <div className="range-labels">
                              <span>{config.min.toFixed(2)}</span>
                              <span>{config.max.toFixed(2)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Preservatives & Alcohol Section */}
                  <div className="form-section">
                    <h3 className="section-title">
                      <Wine size={20} />
                      <span>Preservatives & Alcohol</span>
                    </h3>
                    <div className="form-grid">
                      {['free_sulfur_dioxide', 'total_sulfur_dioxide', 'alcohol'].map(feature => {
                        const config = getFeatureConfig(feature);
                        const value = formData[feature];
                        
                        return (
                          <div key={feature} className="form-group">
                            <label>
                              <span>
                                {getFeatureIcon(feature)}
                                {formatFeatureName(feature)}
                              </span>
                              <span className="label-value">
                                {value.toFixed(1)}
                              </span>
                            </label>
                            <input
                              type="range"
                              min={config.min}
                              max={config.max}
                              step={config.step}
                              value={value}
                              onChange={(e) => handleChange(feature, e.target.value)}
                              aria-label={formatFeatureName(feature)}
                            />
                            <div className="range-labels">
                              <span>{config.min.toFixed(0)}</span>
                              <span>{config.max.toFixed(0)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Preset Buttons */}
                  <div className="preset-buttons">
                    <button 
                      type="button" 
                      className="preset-btn" 
                      onClick={() => loadPreset('Substandard')}
                      aria-label="Load Substandard wine example"
                    >
                      <AlertTriangle size={16} />
                      Substandard Example
                    </button>
                    <button 
                      type="button" 
                      className="preset-btn" 
                      onClick={() => loadPreset('Satisfactory')}
                      aria-label="Load Satisfactory wine example"
                    >
                      <BarChart3 size={16} />
                      Satisfactory Example
                    </button>
                    <button 
                      type="button" 
                      className="preset-btn" 
                      onClick={() => loadPreset('Premium')}
                      aria-label="Load Premium wine example"
                    >
                      <Award size={16} />
                      Premium Example
                    </button>
                  </div>

                  {/* Submit Button */}
                  <button 
                    type="submit" 
                    className="submit-btn" 
                    disabled={loading}
                    aria-label="Predict wine quality"
                  >
                    {loading ? (
                      <>
                        <div className="btn-spinner"></div>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Award size={24} />
                        Predict Wine Quality
                      </>
                    )}
                  </button>
                </form>

                {/* Error Display */}
                {error && (
                  <div 
                    className="error-message"
                    role="alert"
                    aria-live="polite"
                  >
                    <AlertTriangle size={20} />
                    <span>Error: {error}</span>
                  </div>
                )}
              </div>
            </div>

            {/* ========================================
                RESULTS CARD - FIXED: Keep results when error occurs
                ======================================== */}
            <div className="card results-card">
              <div className="card-header">
                <Award size={24} />
                <h2>Quality Assessment</h2>
              </div>

              {/* FIXED: Show empty state only when there's no result AND no loading AND no error */}
              {!result && !loading && (
                <div className="empty-state" aria-live="polite">
                  <div className="empty-icon">
                    <Wine size={64} />
                  </div>
                  <h3>Ready for Analysis</h3>
                  <p>Adjust wine properties and click "Predict Wine Quality" to see results</p>
                </div>
              )}

              {/* Loading State */}
              {loading && (
                <div className="loading-state" aria-live="polite">
                  <div className="spinner" aria-label="Loading results" />
                  <p>Analyzing wine properties...</p>
                </div>
              )}

              {/* Results Display - Show even if there's an error with previous results */}
              {result && (
                <div className="results-content" aria-live="polite">
                  {/* Quality Badge */}
                  <div 
                    className={`quality-badge ${result.quality.toLowerCase()}`}
                    role="status"
                  >
                    <div className="quality-icon">
                      <Wine size={48} />
                    </div>
                    <h3>{result.quality} Quality</h3>
                    <div className="quality-range">
                      {result.quality_score_range}
                    </div>
                    <div className="confidence-badge">
                      {(result.confidence * 100).toFixed(1)}% Confidence
                    </div>
                  </div>

                  {/* Probabilities */}
                  <div className="probability-section">
                    <h4>
                      <BarChart3 size={20} />
                      Quality Probabilities
                    </h4>

                    {QUALITY_ORDER.map((quality) => {
                      const prob = result.probabilities?.[quality] ?? 0;

                      return (
                        <div key={quality} className="probability-bar">
                          <div className="prob-label">
                            <span className="prob-name">{quality}</span>
                            <span className={`prob-value ${quality.toLowerCase()}`}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="prob-track">
                            <div
                              className={`prob-fill ${quality.toLowerCase()}`}
                              style={{ width: `${prob * 100}%` }}
                              aria-label={`${quality}: ${(prob * 100).toFixed(1)}%`}
                            >
                              {prob > 0.15 && `${(prob * 100).toFixed(0)}%`}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Recommendation */}
                  <div className="recommendation-box">
                    <h4>Expert Recommendation</h4>
                    <p>{result.recommendation}</p>
                  </div>

                  {/* Top Factors */}
                  <div className="top-factors">
                    <h4>
                      <TrendingUp size={20} />
                      Top Contributing Factors
                    </h4>
                    {result.top_factors
                      .sort((a, b) => b.importance - a.importance)
                      .map((factor, idx) => (
                      <div key={idx} className="factor-item">
                        <span className="factor-name">
                          {formatFeatureName(factor.feature)}
                        </span>
                        <div className="factor-value">
                          <span className="factor-number">
                            {factor.value.toFixed(2)}
                          </span>
                          <span className="importance-badge">
                            {(factor.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* ========================================
            FOOTER SECTION
            ======================================== */}
        <footer className="footer">
          <p>🍷 Wine Quality Predictor • Powered by Decision Tree ML • Built with FastAPI & React</p>
        </footer>
      </div>
    </div>
  );
}

export default App;