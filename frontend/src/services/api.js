const API_BASE_URL = 'http://127.0.0.1:8080';

export const uploadFiles = async (headerFile, itemsFile, workstationFile) => {
    const formData = new FormData();
    formData.append('header', headerFile);
    formData.append('items', itemsFile);
    formData.append('workstation', workstationFile);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Accept': 'application/json'
            },
            body: formData,
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.error || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Upload error:', error);
        throw error;
    }
};

export const getForecast = async (targets, models, horizon, timePeriod = 'day', aggregationMethod = 'mean', outputFormat = 'json', forDownload = false) => {
    try {
        const payload = {
            targets,
            models,
            horizon,
            time_period: timePeriod,
            aggregation_method: aggregationMethod,
            output_format: outputFormat
        };

        console.log('Sending forecast request:', payload);

        const response = await fetch(`${API_BASE_URL}/forecast`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json',
                'Accept': '*/*'
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage;
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.error || errorData.detail || 'Unknown error';
            } catch {
                errorMessage = errorText || `HTTP error! status: ${response.status}`;
            }
            throw new Error(errorMessage);
        }

        // For data display, return the JSON response directly
        if (!forDownload && outputFormat === 'json') {
            return await response.json();
        }

        // Handle downloads for all formats
        const blob = await (outputFormat === 'json' ? 
            new Blob([JSON.stringify(await response.json(), null, 2)], { type: 'application/json' }) : 
            response.blob());

        const disposition = response.headers.get('Content-Disposition');
        let filename;
        
        if (disposition && disposition.includes('filename=')) {
            filename = disposition.split('filename=')[1].replace(/["']/g, '');
        } else {
            const extension = outputFormat === 'excel' ? 'xlsx' : outputFormat;
            const timestamp = new Date().toISOString().slice(0,19).replace(/[:]/g, '-');
            filename = `forecast_${timePeriod}_${aggregationMethod}_${timestamp}.${extension}`;
        }

        // Create download link and trigger download
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.style.display = 'none';
        link.href = url;
        link.download = filename;
        
        // Add to document, click, and cleanup
        document.body.appendChild(link);
        link.click();
        
        // Cleanup after a short delay
        setTimeout(() => {
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        }, 100);
        
        return true;
    } catch (error) {
        console.error('Forecast error:', error);
        throw error;
    }
};