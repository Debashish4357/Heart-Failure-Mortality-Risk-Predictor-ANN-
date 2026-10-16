import React, { useState, useEffect } from 'react'

const INITIAL_STATE = {
  age: '60',
  anaemia: 0,
  creatinine_phosphokinase: '250',
  diabetes: 0,
  ejection_fraction: '38',
  high_blood_pressure: 0,
  platelets: '260000',
  serum_creatinine: '1.1',
  serum_sodium: '137',
  sex: 1, // 0: Female, 1: Male
  smoking: 0,
}

const PRESET_LOW_RISK = {
  age: '50',
  anaemia: 0,
  creatinine_phosphokinase: '168',
  diabetes: 0,
  ejection_fraction: '45',
  high_blood_pressure: 0,
  platelets: '276000',
  serum_creatinine: '0.9',
  serum_sodium: '140',
  sex: 1,
  smoking: 0,
}

const PRESET_HIGH_RISK = {
  age: '75',
  anaemia: 1,
  creatinine_phosphokinase: '582',
  diabetes: 1,
  ejection_fraction: '20',
  high_blood_pressure: 1,
  platelets: '265000',
  serum_creatinine: '2.5',
  serum_sodium: '130',
  sex: 1,
  smoking: 1,
}

export default function App() {
  const [formData, setFormData] = useState(INITIAL_STATE)
  const [fieldErrors, setFieldErrors] = useState({})
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState(null)
  const [result, setResult] = useState(null)
  const [backendStatus, setBackendStatus] = useState('checking') // 'online' | 'offline' | 'checking'

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
  }, [])

  const checkHealth = async () => {
    try {
      // Try Vite proxy first, then fallback to direct backend
      let res
      try {
        res = await fetch('/api/health')
      } catch {
        res = await fetch('http://127.0.0.1:8000/health')
      }
      if (res && res.ok) {
        setBackendStatus('online')
      } else {
        setBackendStatus('offline')
      }
    } catch {
      setBackendStatus('offline')
    }
  }

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    if (fieldErrors[field]) {
      setFieldErrors((prev) => {
        const next = { ...prev }
        delete next[field]
        return next
      })
    }
  }

  const validate = () => {
    const errors = {}

    const ageNum = parseFloat(formData.age)
    if (isNaN(ageNum) || ageNum <= 0 || ageNum > 120) {
      errors.age = 'Age must be between 1 and 120'
    }

    const cpkNum = parseFloat(formData.creatinine_phosphokinase)
    if (isNaN(cpkNum) || cpkNum <= 0) {
      errors.creatinine_phosphokinase = 'CPK must be a positive number'
    }

    const efNum = parseFloat(formData.ejection_fraction)
    if (isNaN(efNum) || efNum < 1 || efNum > 100) {
      errors.ejection_fraction = 'Ejection fraction must be between 1% and 100%'
    }

    const plateletsNum = parseFloat(formData.platelets)
    if (isNaN(plateletsNum) || plateletsNum <= 0) {
      errors.platelets = 'Platelet count must be a positive number'
    }

    const scNum = parseFloat(formData.serum_creatinine)
    if (isNaN(scNum) || scNum <= 0) {
      errors.serum_creatinine = 'Serum creatinine must be a positive number'
    }

    const ssNum = parseFloat(formData.serum_sodium)
    if (isNaN(ssNum) || ssNum <= 0) {
      errors.serum_sodium = 'Serum sodium must be a positive number'
    }

    setFieldErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setApiError(null)

    if (!validate()) {
      return
    }

    setLoading(true)

    // Build payload with expected types
    const payload = {
      age: parseFloat(formData.age),
      anaemia: parseInt(formData.anaemia, 10),
      creatinine_phosphokinase: parseFloat(formData.creatinine_phosphokinase),
      diabetes: parseInt(formData.diabetes, 10),
      ejection_fraction: parseFloat(formData.ejection_fraction),
      high_blood_pressure: parseInt(formData.high_blood_pressure, 10),
      platelets: parseFloat(formData.platelets),
      serum_creatinine: parseFloat(formData.serum_creatinine),
      serum_sodium: parseFloat(formData.serum_sodium),
      sex: parseInt(formData.sex, 10),
      smoking: parseInt(formData.smoking, 10),
    }

    try {
      let response
      try {
        response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })
      } catch {
        // Fallback directly to localhost FastAPI if proxy is not routing
        response = await fetch('http://127.0.0.1:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(
          errorData.detail ||
            `Server responded with status code ${response.status}: ${response.statusText}`
        )
      }

      const data = await response.json()
      setResult(data)
      setBackendStatus('online')
    } catch (err) {
      setApiError(
        err.message ||
          'Failed to connect to the FastAPI prediction endpoint. Please ensure the backend is running.'
      )
      setBackendStatus('offline')
    } finally {
      setLoading(false)
    }
  }

  const loadPreset = (preset) => {
    setFormData(preset)
    setFieldErrors({})
    setApiError(null)
  }

  // Color helper for risk probability
  const getProbabilityColor = (prob) => {
    if (prob < 0.35) return '#16a34a' // Green
    if (prob < 0.6) return '#d97706' // Amber
    return '#dc2626' // Red
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
            </svg>
          </div>
          <div>
            <h1 className="brand-title">Heart Failure Mortality Predictor</h1>
            <p className="brand-subtitle">
              Clinical decision-support prototype • Powered by FastAPI & Scikit-Learn
            </p>
          </div>
        </div>

        <div className="header-status" title="FastAPI Backend Status">
          <span
            className={`status-dot ${
              backendStatus === 'online'
                ? 'online'
                : backendStatus === 'offline'
                ? 'offline'
                : ''
            }`}
          />
          <span>
            {backendStatus === 'online'
              ? 'FastAPI Connected'
              : backendStatus === 'offline'
              ? 'Backend Disconnected'
              : 'Checking API...'}
          </span>
          <button
            onClick={checkHealth}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--text-muted)',
              fontSize: '0.75rem',
              textDecoration: 'underline',
              marginLeft: '0.25rem',
            }}
          >
            Retry
          </button>
        </div>
      </header>

      {/* Quick Fill / Presets */}
      <div className="presets-bar">
        <span className="presets-label">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          Quick Presets:
        </span>
        <button
          type="button"
          className="preset-btn"
          onClick={() => loadPreset(PRESET_LOW_RISK)}
        >
          Normal Risk Profile
        </button>
        <button
          type="button"
          className="preset-btn"
          onClick={() => loadPreset(PRESET_HIGH_RISK)}
        >
          High Risk Profile
        </button>
        <button
          type="button"
          className="preset-btn btn-reset"
          onClick={() => loadPreset(INITIAL_STATE)}
        >
          Reset Values
        </button>
      </div>

      {/* Error alert banner */}
      {apiError && (
        <div className="alert-banner error" role="alert">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div>
            <strong>Prediction API Error:</strong>
            <div>{apiError}</div>
            <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', opacity: 0.9 }}>
              Tip: Verify that the FastAPI backend is running via{' '}
              <code>uvicorn main:app --reload</code> or <code>python main.py</code> at{' '}
              <code>http://127.0.0.1:8000</code>.
            </div>
          </div>
        </div>
      )}

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Input Form Card */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
              Patient Clinical Parameters (11 Features)
            </h2>
          </div>

          <div className="card-body">
            <form onSubmit={handleSubmit}>
              {/* Demographics & Physical */}
              <div className="form-section-title">Demographics & Vital Metrics</div>
              <div className="form-grid">
                {/* Age */}
                <div className="form-group">
                  <label className="form-label" htmlFor="age">
                    <span>Age</span>
                    <span className="label-unit">years</span>
                  </label>
                  <input
                    id="age"
                    type="number"
                    step="1"
                    min="1"
                    max="120"
                    className={`form-input ${fieldErrors.age ? 'has-error' : ''}`}
                    value={formData.age}
                    onChange={(e) => handleInputChange('age', e.target.value)}
                    required
                  />
                  {fieldErrors.age && (
                    <span className="error-text">{fieldErrors.age}</span>
                  )}
                </div>

                {/* Biological Sex */}
                <div className="form-group">
                  <label className="form-label">
                    <span>Biological Sex</span>
                  </label>
                  <div className="toggle-group">
                    <button
                      type="button"
                      className={`toggle-option ${formData.sex === 0 ? 'active' : ''}`}
                      onClick={() => handleInputChange('sex', 0)}
                    >
                      Female (0)
                    </button>
                    <button
                      type="button"
                      className={`toggle-option ${formData.sex === 1 ? 'active' : ''}`}
                      onClick={() => handleInputChange('sex', 1)}
                    >
                      Male (1)
                    </button>
                  </div>
                </div>

                {/* Ejection Fraction */}
                <div className="form-group">
                  <label className="form-label" htmlFor="ejection_fraction">
                    <span>Ejection Fraction</span>
                    <span className="label-unit">% of blood leaving heart</span>
                  </label>
                  <input
                    id="ejection_fraction"
                    type="number"
                    step="1"
                    min="1"
                    max="100"
                    className={`form-input ${
                      fieldErrors.ejection_fraction ? 'has-error' : ''
                    }`}
                    value={formData.ejection_fraction}
                    onChange={(e) =>
                      handleInputChange('ejection_fraction', e.target.value)
                    }
                    required
                  />
                  {fieldErrors.ejection_fraction && (
                    <span className="error-text">{fieldErrors.ejection_fraction}</span>
                  )}
                </div>
              </div>

              {/* Lab / Serum Metrics */}
              <div className="form-section-title">Laboratory Blood Tests</div>
              <div className="form-grid">
                {/* Serum Creatinine */}
                <div className="form-group">
                  <label className="form-label" htmlFor="serum_creatinine">
                    <span>Serum Creatinine</span>
                    <span className="label-unit">mg/dL</span>
                  </label>
                  <input
                    id="serum_creatinine"
                    type="number"
                    step="0.01"
                    min="0.1"
                    className={`form-input ${
                      fieldErrors.serum_creatinine ? 'has-error' : ''
                    }`}
                    value={formData.serum_creatinine}
                    onChange={(e) =>
                      handleInputChange('serum_creatinine', e.target.value)
                    }
                    required
                  />
                  {fieldErrors.serum_creatinine && (
                    <span className="error-text">
                      {fieldErrors.serum_creatinine}
                    </span>
                  )}
                </div>

                {/* Serum Sodium */}
                <div className="form-group">
                  <label className="form-label" htmlFor="serum_sodium">
                    <span>Serum Sodium</span>
                    <span className="label-unit">mEq/L</span>
                  </label>
                  <input
                    id="serum_sodium"
                    type="number"
                    step="1"
                    min="50"
                    max="200"
                    className={`form-input ${
                      fieldErrors.serum_sodium ? 'has-error' : ''
                    }`}
                    value={formData.serum_sodium}
                    onChange={(e) =>
                      handleInputChange('serum_sodium', e.target.value)
                    }
                    required
                  />
                  {fieldErrors.serum_sodium && (
                    <span className="error-text">{fieldErrors.serum_sodium}</span>
                  )}
                </div>

                {/* Creatinine Phosphokinase (CPK) */}
                <div className="form-group">
                  <label className="form-label" htmlFor="creatinine_phosphokinase">
                    <span>Creatinine Phosphokinase</span>
                    <span className="label-unit">mcg/L</span>
                  </label>
                  <input
                    id="creatinine_phosphokinase"
                    type="number"
                    step="1"
                    min="1"
                    className={`form-input ${
                      fieldErrors.creatinine_phosphokinase ? 'has-error' : ''
                    }`}
                    value={formData.creatinine_phosphokinase}
                    onChange={(e) =>
                      handleInputChange('creatinine_phosphokinase', e.target.value)
                    }
                    required
                  />
                  {fieldErrors.creatinine_phosphokinase && (
                    <span className="error-text">
                      {fieldErrors.creatinine_phosphokinase}
                    </span>
                  )}
                </div>

                {/* Platelets */}
                <div className="form-group">
                  <label className="form-label" htmlFor="platelets">
                    <span>Platelets</span>
                    <span className="label-unit">kiloplatelets/mL</span>
                  </label>
                  <input
                    id="platelets"
                    type="number"
                    step="1000"
                    min="1000"
                    className={`form-input ${
                      fieldErrors.platelets ? 'has-error' : ''
                    }`}
                    value={formData.platelets}
                    onChange={(e) => handleInputChange('platelets', e.target.value)}
                    required
                  />
                  {fieldErrors.platelets && (
                    <span className="error-text">{fieldErrors.platelets}</span>
                  )}
                </div>
              </div>

              {/* Comorbidities / History */}
              <div className="form-section-title">Comorbidities & Risk Factors</div>
              <div className="form-grid">
                {/* Anaemia */}
                <div className="form-group">
                  <label className="form-label">
                    <span>Anaemia</span>
                    <span className="label-unit">RBC decrease</span>
                  </label>
                  <div className="toggle-group">
                    <button
                      type="button"
                      className={`toggle-option ${
                        formData.anaemia === 0 ? 'active' : ''
                      }`}
                      onClick={() => handleInputChange('anaemia', 0)}
                    >
                      No (0)
                    </button>
                    <button
                      type="button"
                      className={`toggle-option ${
                        formData.anaemia === 1 ? 'active' : ''
                      }`}
                      onClick={() => handleInputChange('anaemia', 1)}
                    >
                      Yes (1)
                    </button>
                  </div>
                </div>

                {/* Diabetes */}
                <div className="form-group">
                  <label className="form-label">
                    <span>Diabetes</span>
                    <span className="label-unit">History</span>
                  </label>
                  <div className="toggle-group">
                    <button
                      type="button"
                      className={`toggle-option ${
                        formData.diabetes === 0 ? 'active' : ''
                      }`}
                      onClick={() => handleInputChange('diabetes', 0)}
                    >
                      No (0)
                    </button>
                    <button
                      type="button"
                      className={`toggle-option ${
                        formData.diabetes === 1 ? 'active' : ''
                      }`}
                      onClick={() => handleInputChange('diabetes', 1)}
                    >
                      Yes (1)
                    </button>
                  </div>
                </div>

                {/* High Blood Pressure */}
                <div className="form-group">
                  <label className="form-label">
                    <span>High Blood Pressure</span>
                    <span className="label-unit">Hypertension</span>
                  </label>
                  <div className="toggle-group">
                    <button
                      type="button"
                      className={`toggle-option ${
                        formData.high_blood_pressure === 0 ? 'active' : ''
                      }`}
                      onClick={() => handleInputChange('high_blood_pressure', 0)}
                    >
                      No (0)
                    </button>
                    <button
                      type="button"
                      className={`toggle-option ${
                        formData.high_blood_pressure === 1 ? 'active' : ''
                      }`}
                      onClick={() => handleInputChange('high_blood_pressure', 1)}
                    >
                      Yes (1)
                    </button>
                  </div>
                </div>

                {/* Smoking */}
                <div className="form-group">
                  <label className="form-label">
                    <span>Smoking Status</span>
                    <span className="label-unit">Tobacco use</span>
                  </label>
                  <div className="toggle-group">
                    <button
                      type="button"
                      className={`toggle-option ${
                        formData.smoking === 0 ? 'active' : ''
                      }`}
                      onClick={() => handleInputChange('smoking', 0)}
                    >
                      No (0)
                    </button>
                    <button
                      type="button"
                      className={`toggle-option ${
                        formData.smoking === 1 ? 'active' : ''
                      }`}
                      onClick={() => handleInputChange('smoking', 1)}
                    >
                      Yes (1)
                    </button>
                  </div>
                </div>
              </div>

              {/* Form Actions */}
              <div className="form-actions">
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={loading}
                  id="predict-button"
                >
                  {loading ? (
                    <>
                      <div className="spinner" />
                      <span>Computing Risk...</span>
                    </>
                  ) : (
                    <>
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                      </svg>
                      <span>Predict Mortality Risk</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Prediction Results Card */}
        <div className="card results-card">
          <div className="card-header">
            <h2 className="card-title">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 14 14" />
              </svg>
              Prediction Output
            </h2>
          </div>

          <div className="card-body">
            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">
                  <svg
                    width="28"
                    height="28"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.75"
                  >
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                  </svg>
                </div>
                <h3>No Prediction Yet</h3>
                <p>
                  Fill out the 11 clinical features on the left or click a preset, then
                  click "Predict Mortality Risk".
                </p>
              </div>
            ) : (
              <div className="prediction-result">
                {/* Result Classification Badge */}
                <div
                  className={`status-badge-card ${
                    result.predicted_class === 1 ? 'high-risk' : 'low-risk'
                  }`}
                >
                  <div className="status-badge-icon">
                    {result.predicted_class === 1 ? (
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
                        <line x1="12" y1="8" x2="12" y2="12" />
                        <line x1="12" y1="16" x2="12.01" y2="16" />
                      </svg>
                    ) : (
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                        <polyline points="22 4 12 14.01 9 11.01" />
                      </svg>
                    )}
                  </div>
                  <div className="status-badge-content">
                    <h4>Model Assessment</h4>
                    <div className="label">{result.predicted_label}</div>
                  </div>
                </div>

                {/* Probability Gauge */}
                <div className="metric-container">
                  <div className="metric-header">
                    <span className="metric-title">Estimated Mortality Risk</span>
                    <span
                      className="metric-value"
                      style={{
                        color: getProbabilityColor(result.predicted_probability),
                      }}
                    >
                      {(result.predicted_probability * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="progress-bar-bg">
                    <div
                      className="progress-bar-fill"
                      style={{
                        width: `${Math.min(
                          Math.max(result.predicted_probability * 100, 2),
                          100
                        )}%`,
                        backgroundColor: getProbabilityColor(
                          result.predicted_probability
                        ),
                      }}
                    />
                  </div>

                  <div className="probability-scale">
                    <span>0% (Low Risk)</span>
                    <span>
                      Raw Probability: {result.predicted_probability.toFixed(4)}
                    </span>
                    <span>100% (High Risk)</span>
                  </div>
                </div>

                {/* API Notice */}
                <div className="disclaimer-box">
                  <strong>Notice:</strong> {result.notice}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="app-footer">
        <p>Heart Failure Mortality Risk Predictor • Educational ML Decision-Support Prototype</p>
      </footer>
    </div>
  )
}
