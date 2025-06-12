const API_BASE_URL = 'http://localhost:8080';

const AVAILABLE_MODELS = {
    ARIMA: 'ARIMA',
    Prophet: 'Prophet',
    LSTM: 'LSTM',
    RandomForest: 'RandomForest',
    EMA: 'EMA',
    HoltWinters: 'HoltWinters'
};

const VALID_TIME_PERIODS = ['day', 'week', 'month'];
const VALID_AGG_METHODS = ['mean', 'sum', 'min', 'max'];

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
        // Validate and sanitize inputs
        const sanitizedHorizon = parseInt(horizon);
        if (isNaN(sanitizedHorizon) || sanitizedHorizon <= 0) {
            throw new Error('Invalid horizon value');
        }

        if (!VALID_TIME_PERIODS.includes(timePeriod)) {
            throw new Error('Invalid time period');
        }

        if (!VALID_AGG_METHODS.includes(aggregationMethod)) {
            throw new Error('Invalid aggregation method');
        }

        console.log('Starting forecast request with:', { targets, models, horizon: sanitizedHorizon, timePeriod, aggregationMethod });

        const targetList = Array.isArray(targets) ? targets : [targets];
        let formattedModels = {};

        if (typeof models === 'object' && models !== null) {
            // Handle both string model types and model info objects
            targetList.forEach(target => {
                const model = models[target];
                if (model) {
                    formattedModels[target] = typeof model === 'string' ? model : (model.id || model.name);
                } else {
                    formattedModels[target] = 'Prophet';
                }
            });
        }        const payload = {
            targets: targetList,
            models: formattedModels,
            horizon: sanitizedHorizon,
            time_period: timePeriod,
            aggregation_method: aggregationMethod,
            output_format: outputFormat.toLowerCase(),
            forDownload: forDownload
        };

        console.log('Sending forecast request:', payload);
        
        const response = await fetch(`${API_BASE_URL}/forecast`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json',
                'Accept': forDownload ? '*/*' : 'application/json'
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage;
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.detail && Array.isArray(errorData.detail)
                    ? errorData.detail.map(err => `${err.loc.join('.')}: ${err.msg}`).join(', ')
                    : (errorData.error || errorData.detail || 'Unknown error');
            } catch {
                errorMessage = errorText || `HTTP error! status: ${response.status}`;
            }
            throw new Error(errorMessage);
        }

        // For data display without download, parse JSON
        if (!forDownload) {
            return await response.json();
        }

        // For downloads, handle each format appropriately
        let blob;
        if (outputFormat === 'json') {
            // For JSON, use the response directly as it's already filtered
            const data = await response.json();
            blob = new Blob([JSON.stringify(data, null, 2)], { 
                type: 'application/json' 
            });
        } else if (outputFormat === 'excel') {
            // For Excel, use the raw response
            const buffer = await response.arrayBuffer();
            blob = new Blob([buffer], { 
                type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            });
        } else {
            // For CSV, use the raw response
            const text = await response.text();
            blob = new Blob([text], { 
                type: 'text/csv'
            });
        }

        // Get filename from Content-Disposition header or generate one
        const disposition = response.headers.get('Content-Disposition');
        let filename;
        if (disposition && disposition.includes('filename=')) {
            filename = disposition.split('filename=')[1].replace(/["']/g, '');
        } else {
            const extension = outputFormat === 'excel' ? 'xlsx' : outputFormat;
            const timestamp = new Date().toISOString().slice(0,19).replace(/[:]/g, '-');
            const targetNames = targetList.join('_');
            filename = `forecast_${targetNames}_${timePeriod}_${aggregationMethod}_${timestamp}.${extension}`;
        }

        // Create download link and trigger download
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.style.display = 'none';
        link.href = url;
        link.download = filename;
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