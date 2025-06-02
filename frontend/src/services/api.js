const API_BASE_URL = 'http://127.0.0.1:8000';

export const uploadFiles = async (headerFile, itemsFile, workstationFile) => {
    const formData = new FormData();
    formData.append('header', headerFile);
    formData.append('items', itemsFile);
    formData.append('workstation', workstationFile);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Upload error:', error);
        throw error;
    }
};

export const getForecast = async (targets, models, horizon, outputFormat = 'json') => {
    try {
        const response = await fetch(`${API_BASE_URL}/forecast`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                targets,
                models,
                horizon,
                output_format: outputFormat
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Handle different response types
        if (outputFormat === 'json') {
            return await response.json();
        } else if (outputFormat === 'csv' || outputFormat === 'excel') {
            const blob = await response.blob();
            const fileName = `forecast.${outputFormat}`;
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            return true;
        }
    } catch (error) {
        console.error('Forecast error:', error);
        throw error;
    }
};