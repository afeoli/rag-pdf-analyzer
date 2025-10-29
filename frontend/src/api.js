const API_BASE_URL = 'http://localhost:8000';

export const api = {
  async uploadPdf(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }
    
    return response.json();
  },
  
  async queryDocuments(question) {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Query failed');
    }
    
    return response.json();
  },
  
  async checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },
  
  async getConfiguration() {
    const response = await fetch(`${API_BASE_URL}/configuration`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get configuration');
    }
    
    return response.json();
  },
  
  async updateConfiguration(configData) {
    const response = await fetch(`${API_BASE_URL}/configuration`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(configData),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update configuration');
    }
    
    return response.json();
  }
};
