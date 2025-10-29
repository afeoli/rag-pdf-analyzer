import { useState } from 'react';
import { api } from './api';
import './App.css';

function App() {
  const [apiKey, setApiKey] = useState('');
  const [apiKeySet, setApiKeySet] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [confidenceScore, setConfidenceScore] = useState(null);
  const [sources, setSources] = useState([]);
  const [extractedText, setExtractedText] = useState('');
  const [hasDocuments, setHasDocuments] = useState(false);

  const [configuration, setConfiguration] = useState(null);
  const [showConfiguration, setShowConfiguration] = useState(false);
  const [configForm, setConfigForm] = useState({
    chunk_size: 1000,
    chunk_overlap: 200,
    temperature: 0.0,
    model_name: 'gpt-3.5-turbo'
  });

  const handleSetApiKey = async () => {
    if (!apiKey.trim()) {
      alert('Please enter a valid API key');
      return;
    }
    
    // Validate API key by making a test call
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/set-api-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ api_key: apiKey }),
      });
      
      if (response.ok) {
        setApiKeySet(true);
        alert('API key set successfully');
      } else {
        const error = await response.json();
        alert(`Failed to set API key: ${error.detail || 'Invalid API key'}`);
      }
    } catch (error) {
      alert(`Error setting API key: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGetConfiguration = async () => {
    setIsLoading(true);

    try {
      const result = await api.getConfiguration();
      setConfiguration(result);
      setConfigForm(result.configuration);
      setShowConfiguration(true);
    } catch (error) {
      setConfiguration({ error: error.message });
      setShowConfiguration(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateConfiguration = async () => {
    setIsLoading(true);

    try {
      const result = await api.updateConfiguration(configForm);
      setConfiguration(result);
      setUploadStatus(`Configuration updated successfully!`);
    } catch (error) {
      setUploadStatus(`Error updating configuration: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfigFormChange = (field, value) => {
    setConfigForm(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // Helper functions for confidence score display
  const getConfidenceLevel = (score) => {
    if (score >= 80) return 'high';
    if (score >= 60) return 'medium';
    if (score >= 40) return 'low';
    return 'very-low';
  };

  const getConfidenceDescription = (score) => {
    if (score >= 80) return 'High confidence - Answer is very reliable based on PDF content';
    if (score >= 60) return 'Medium confidence - Answer is reasonably reliable from PDF';
    if (score >= 40) return 'Low confidence - Answer may be incomplete or uncertain from PDF';
    return 'Very low confidence - Answer is unreliable or insufficient from PDF';
  };

  const cleanExtractedText = (text) => {
    if (!text) return '';
    
    let cleaned = text;
    
    const patterns = [
      /^I will provide the complete text[:\s]*/i,
      /^Here is the extracted text[:\s]*/i,
      /^I'll extract.*?[\n\r]+/i,
      /^I can.*?extract.*?[\n\r]+/i,
      /^Here.*?extracted.*?[\n\r]+/i,
      /^The.*?content.*?[\n\r]+/i,
      /^Below.*?extracted.*?[\n\r]+/i,
    ];
    
    for (const pattern of patterns) {
      cleaned = cleaned.replace(pattern, '');
    }
    
    cleaned = cleaned.trim();
    
    if (cleaned.toLowerCase().startsWith('here') || cleaned.toLowerCase().startsWith('the following')) {
      const lines = cleaned.split('\n');
      if (lines.length > 1) {
        cleaned = lines.slice(1).join('\n').trim();
      }
    }
    
    return cleaned;
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const isPdf = selectedFile.type === 'application/pdf' || selectedFile.name.toLowerCase().endsWith('.pdf');
      if (isPdf) {
        setFile(selectedFile);
        setUploadStatus('');
      } else {
        setFile(null);
        setUploadStatus('Please select a valid PDF file');
      }
    } else {
      setFile(null);
      setUploadStatus('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus('Please select a file first');
      return;
    }

    setIsLoading(true);
    setUploadStatus('');
    
    // console.log('Uploading file:', file.name, file.size);

    try {
      const result = await api.uploadPdf(file);
      const cleanedText = cleanExtractedText(result.extracted_text || '');
      setUploadStatus(`Document processed successfully (${result.chunks} chunks)`);
      setExtractedText(cleanedText);
      setHasDocuments(true);
      
      const fileInput = document.getElementById('file-input');
      if (fileInput) {
        fileInput.value = '';
      }
      setFile(null);
    } catch (error) {
      setUploadStatus(`Error: ${error.message}`);
      setExtractedText('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!question.trim()) {
      setAnswer('Please enter a question');
      return;
    }

    if (!hasDocuments) {
      setAnswer('Please upload a PDF document first');
      return;
    }

    setIsLoading(true);
    setAnswer('');

    try {
      const result = await api.queryDocuments(question);
      setAnswer(result.answer);
      setConfidenceScore(result.confidence_score);
      setSources(result.sources || []);
    } catch (error) {
      setAnswer(`Error: ${error.message}`);
      setConfidenceScore(null);
      setSources([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };


  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <h1>RAG PDF Analyzer</h1>
            <p>Upload a PDF file and ask questions about it</p>
          </div>
          <div className="header-right">
            <button 
              onClick={handleGetConfiguration} 
              disabled={isLoading || !apiKeySet}
              className="config-btn header-config-btn"
            >
              {isLoading ? 'Loading' : 'Configuration'}
            </button>
          </div>
        </div>
      </header>

      <main className="app-main">
        {/* API Key Section */}
        {!apiKeySet && (
          <section className="api-key-section">
            <h2>OpenAI API Key</h2>
            <div className="api-key-area">
              <p>Please enter your OpenAI API key to continue</p>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="api-key-input"
              />
              <button 
                onClick={handleSetApiKey} 
                disabled={!apiKey.trim() || isLoading}
                className="api-key-btn"
              >
                {isLoading ? 'Setting...' : 'Set API Key'}
              </button>
            </div>
          </section>
        )}

        {apiKeySet && (
          <>
            {/* Query Section */}
            {hasDocuments && (
              <section className="query-section">
                <h2>Ask Questions</h2>
                <div className="query-area">
                  <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask a question about the uploaded document..."
                    className="question-input"
                    rows="3"
                  />
                  <div className="query-buttons">
                    <button 
                      onClick={handleQuery} 
                      disabled={!apiKeySet || !hasDocuments || isLoading || !question.trim()}
                      className="query-btn"
                    >
                      {isLoading ? 'Processing' : 'Query'}
                    </button>
                  </div>
                </div>
                
                {answer && (
                  <div className="answer-area">
              <div className="answer-header">
                <h3>Answer</h3>
                {confidenceScore !== null && (
                  <div className="confidence-indicator">
                    <div className={`confidence-badge ${getConfidenceLevel(confidenceScore)}`}>
                      {confidenceScore}/100
                    </div>
                    <span className="confidence-label">Confidence</span>
                  </div>
                )}
              </div>
              
                    <div className="answer-content">
                      <div className="answer-text">{answer}</div>
                      
                      {confidenceScore !== null && (
                        <div className="confidence-details">
                          <div className="confidence-bar">
                            <div className={`confidence-fill ${getConfidenceLevel(confidenceScore)}`} 
                                 style={{width: `${confidenceScore}%`}}></div>
                          </div>
                          <div className="confidence-description">
                            {getConfidenceDescription(confidenceScore)}
                          </div>
                          
                          {sources && sources.length > 0 && (
                            <div className="sources-section">
                              <h4>Sources:</h4>
                              <div className="sources-list">
                                {sources.map((source, idx) => (
                                  <div key={idx} className="source-item">
                                    <div className="source-meta">
                                      <span className="source-page">Page {source.page}</span>
                                      {source.section !== 'N/A' && (
                                        <span className="source-section">• {source.section}</span>
                                      )}
                                    </div>
                                    <div className="source-preview">{source.preview}</div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </section>
            )}

        {/* Upload Section */}
        <section className="upload-section">
          <div className="upload-header">
            <h2>Upload PDF</h2>
          </div>
          <div className="upload-area">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="file-input"
              id="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              {file ? file.name : 'Select PDF file'}
            </label>
            <button 
              onClick={handleUpload} 
              disabled={!apiKeySet || !file || isLoading}
              className="upload-btn"
              title={!apiKeySet ? 'API key required' : !file ? 'Select a PDF file' : isLoading ? 'Processing...' : ''}
            >
              {isLoading ? 'Processing...' : hasDocuments ? 'Replace Document' : 'Upload'}
            </button>
          </div>
          {uploadStatus && (
            <div className={`upload-message ${uploadStatus.includes('Error') || uploadStatus.includes('error') ? 'error' : 'success'}`}>
              {uploadStatus.replace(/✅/g, '').trim()}
            </div>
          )}
          {extractedText && (
            <div className="extracted-text-section">
              <h3>Extracted Text:</h3>
              <div className="extracted-text-content">
                {extractedText}
              </div>
            </div>
          )}
        </section>

            {/* Configuration Display Section */}
            {showConfiguration && configuration && (
          <section className="configuration-display-section">
            <div className="configuration-display-header">
              <h2>RAG Configuration</h2>
              <button 
                onClick={() => setShowConfiguration(false)}
                className="close-btn"
              >
                Close
              </button>
            </div>
            <div className="configuration-display-content">
              {configuration.error ? (
                <div className="error-message">Error: {configuration.error}</div>
              ) : (
                <div>
                  <div className="current-config">
                    <h3>Current Configuration</h3>
                    <div className="config-grid">
                      <div className="config-item">
                        <label>Chunk Size:</label>
                        <span>{configuration.configuration?.chunk_size || configForm.chunk_size} characters</span>
                      </div>
                      <div className="config-item">
                        <label>Chunk Overlap:</label>
                        <span>{configuration.configuration?.chunk_overlap || configForm.chunk_overlap} characters</span>
                      </div>
                      <div className="config-item">
                        <label>Temperature:</label>
                        <span>{configuration.configuration?.temperature || configForm.temperature}</span>
                      </div>
                      <div className="config-item">
                        <label>Model:</label>
                        <span>{configuration.configuration?.model_name || configForm.model_name}</span>
                      </div>
                    </div>
                  </div>

                  <div className="config-form">
                    <h3>Update</h3>
                    <div className="form-grid">
                      <div className="form-group">
                        <label htmlFor="chunk_size">Chunk Size (100-5000):</label>
                        <input
                          type="number"
                          id="chunk_size"
                          value={configForm.chunk_size}
                          onChange={(e) => handleConfigFormChange('chunk_size', parseInt(e.target.value))}
                          min="100"
                          max="5000"
                          className="config-input"
                        />
                      </div>
                      <div className="form-group">
                        <label htmlFor="chunk_overlap">Chunk Overlap (0-{Math.floor(configForm.chunk_size / 2)}):</label>
                        <input
                          type="number"
                          id="chunk_overlap"
                          value={configForm.chunk_overlap}
                          onChange={(e) => handleConfigFormChange('chunk_overlap', parseInt(e.target.value))}
                          min="0"
                          max={Math.floor(configForm.chunk_size / 2)}
                          className="config-input"
                        />
                      </div>
                      <div className="form-group">
                        <label htmlFor="temperature">Temperature (0.0-2.0):</label>
                        <input
                          type="number"
                          id="temperature"
                          value={configForm.temperature}
                          onChange={(e) => handleConfigFormChange('temperature', parseFloat(e.target.value))}
                          min="0.0"
                          max="2.0"
                          step="0.1"
                          className="config-input"
                        />
                      </div>
                      <div className="form-group">
                        <label htmlFor="model_name">Model:</label>
                        <select
                          id="model_name"
                          value={configForm.model_name}
                          onChange={(e) => handleConfigFormChange('model_name', e.target.value)}
                          className="config-select"
                        >
                          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                          <option value="gpt-4">GPT-4</option>
                          <option value="gpt-4-turbo">GPT-4 Turbo</option>
                        </select>
                      </div>
                    </div>
                    <button 
                      onClick={handleUpdateConfiguration}
                      disabled={isLoading}
                      className="update-config-btn"
                    >
                      {isLoading ? 'Updating' : 'Update'}
                    </button>
                  </div>

                  {configuration.explanations && (
                    <div className="parameter-explanations">
                      <h3>Parameter Notes</h3>
                      {Object.entries(configuration.explanations).map(([key, explanation]) => (
                        <div key={key} className="explanation-item">
                          <h4>{key.replace('_', ' ')}</h4>
                          <p><strong>Current:</strong> {explanation.current}</p>
                          <p>{explanation.description}</p>
                          <p className="note-text">{explanation.note}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        )}
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by OpenAI • Built with FastAPI & React</p>
      </footer>
    </div>
  );
}

export default App;