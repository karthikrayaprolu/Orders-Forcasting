import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useWorkflow } from '@/contexts/WorkflowContext';
import { Button } from '@/components/ui/button';
import { FiDownload } from 'react-icons/fi';
import { getForecast } from '@/services/api';
import { toast } from 'sonner';
import Plot from 'react-plotly.js';
import ErrorBoundary from '@/components/ErrorBoundary';

const Results = () => {
    const { results, setResults } = useWorkflow();
    const [selectedFormat, setSelectedFormat] = useState('json');
    const [downloading, setDownloading] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [targetGraphs, setTargetGraphs] = useState([]);

    useEffect(() => {
        // Load saved configurations and results
        const storedSelections = localStorage.getItem('modelSelections');
        const storedTargets = localStorage.getItem('selectedTargets');
        const storedResults = localStorage.getItem('forecastResults');
        const horizon = parseInt(localStorage.getItem('forecastHorizon')) || 30;
        const timePeriod = localStorage.getItem('timePeriod') || 'day';
        const aggregationMethod = localStorage.getItem('aggregationMethod') || 'mean';
        
        // First try to load from localStorage
        if (!results && storedResults) {
            try {
                const parsedResults = JSON.parse(storedResults);
                setResults(parsedResults);
            } catch (e) {
                console.error('Failed to parse stored results:', e);
                toast.error('Failed to load stored results');
            }
        }

        // If no stored results or parsing failed, reload from API
        if (!results && storedSelections && storedTargets) {
            setLoading(true);
            const modelSelections = JSON.parse(storedSelections);
            const targets = JSON.parse(storedTargets);
            
            getForecast(targets, modelSelections, horizon, timePeriod, aggregationMethod, 'json', false)
                .then(data => {
                    if (data && !data.error) {
                        setResults(data);
                        localStorage.setItem('forecastResults', JSON.stringify(data));
                    } else {
                        throw new Error(data?.error || 'Failed to load forecast data');
                    }
                })
                .catch(error => {
                    console.error('Error reloading forecast:', error);
                    setError(`Failed to load forecast: ${error.message}`);
                    toast.error(`Failed to load forecast: ${error.message}`);
                })
                .finally(() => setLoading(false));
        }
    }, []);

    useEffect(() => {
        if (results && results.length > 0) {
            const targets = Object.keys(results[0]).filter(key => key !== 'date');
            
            // Get the earliest and latest dates from the data
            const dates = results.map(r => new Date(r.date));
            const minDate = new Date(Math.min(...dates));
            const maxDate = new Date(Math.max(...dates));

            // The split date should be where historical data ends and forecast begins
            // This is typically the last date before the forecast horizon starts
            // We'll find it by looking at the date pattern in the data
            let splitDate = null;
            const timeDiffs = [];
            for (let i = 1; i < dates.length; i++) {
                timeDiffs.push(dates[i].getTime() - dates[i-1].getTime());
            }
            
            // Find where the time difference pattern changes - this is likely where forecast starts
            const medianDiff = timeDiffs.slice(0, Math.floor(timeDiffs.length / 2)).reduce((a, b) => a + b, 0) / Math.floor(timeDiffs.length / 2);
            let splitIndex = timeDiffs.findIndex(diff => Math.abs(diff - medianDiff) > medianDiff * 0.1);
            if (splitIndex === -1) {
                // If no clear split is found, use the middle point
                splitIndex = Math.floor(dates.length / 2);
            }
            splitDate = dates[splitIndex];

            console.log('Date analysis:', {
                minDate,
                maxDate,
                splitDate,
                totalPoints: dates.length,
                historicalPoints: splitIndex + 1,
                forecastPoints: dates.length - splitIndex - 1
            });
            
            // Process data for each target
            const processedGraphs = targets.map(target => {
                // Ensure dates are properly parsed
                const processedResults = results.map(r => ({
                    ...r,
                    date: new Date(r.date)
                }));

                // Split data into historical and forecast periods
                const historicalData = processedResults.filter((r, i) => i <= splitIndex);
                const forecastData = processedResults.filter((r, i) => i > splitIndex);
                
                // Calculate y-axis range including both historical and forecast data
                const allValues = results.map(r => r[target]);
                const minValue = Math.min(...allValues) * 0.9;
                const maxValue = Math.max(...allValues) * 1.1;

                // Calculate confidence intervals for forecast (±10%)
                const confidenceUpper = forecastData.map(d => ({
                    date: d.date,
                    value: d[target] * 1.1
                }));
                const confidenceLower = forecastData.map(d => ({
                    date: d.date,
                    value: d[target] * 0.9
                }));

                return {
                    target,
                    traces: [
                        // Historical data trace
                        {
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Historical Data',
                            x: historicalData.map(d => d.date),
                            y: historicalData.map(d => d[target]),
                            line: { 
                                color: '#4f46e5',
                                width: 2.5
                            },
                            marker: {
                                size: 6,
                                color: '#4f46e5'
                            },
                            hovertemplate: 'Value: %{y:.2f}<br>Date: %{x|%B %d, %Y}<extra>Historical</extra>'
                        },
                        // Forecast data trace
                        {
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Forecast',
                            x: forecastData.map(d => d.date),
                            y: forecastData.map(d => d[target]),
                            line: { 
                                color: '#f59e0b',
                                width: 3,
                                dash: 'dash'
                            },
                            hovertemplate: 'Value: %{y:.2f}<br>Date: %{x|%B %d, %Y}<extra>Forecast</extra>'
                        },
                        // Confidence interval
                        {
                            type: 'scatter',
                            name: 'Confidence Interval (±10%)',
                            x: [...forecastData.map(d => d.date), ...forecastData.map(d => d.date).reverse()],
                            y: [...confidenceUpper.map(d => d.value), ...confidenceLower.map(d => d.value).reverse()],
                            fill: 'toself',
                            fillcolor: 'rgba(245, 158, 11, 0.1)',
                            line: { width: 0 },
                            showlegend: true,
                            hoverinfo: 'skip'
                        }
                    ],
                    layout: {
                        title: {
                            text: `Forecast for ${target}`,
                            font: { size: 24, color: '#1f2937' }
                        },
                        xaxis: {
                            title: 'Date',
                            showgrid: true,
                            gridcolor: '#f3f4f6',
                            rangeslider: { visible: true },
                            rangeselector: {
                                buttons: [
                                    {count: 1, label: '1m', step: 'month', stepmode: 'backward'},
                                    {count: 6, label: '6m', step: 'month', stepmode: 'backward'},
                                    {count: 1, label: '1y', step: 'year', stepmode: 'backward'},
                                    {step: 'all', label: 'All'}
                                ],
                                bgcolor: '#f9fafb',
                                activecolor: '#4f46e5'
                            }
                        },
                        yaxis: {
                            title: 'Value',
                            showgrid: true,
                            gridcolor: '#f3f4f6',
                            range: [minValue, maxValue]
                        },
                        shapes: [{
                            type: 'line',
                            x0: splitDate,
                            x1: splitDate,
                            y0: minValue,
                            y1: maxValue,
                            line: {
                                color: '#ef4444',
                                width: 1,
                                dash: 'dash'
                            }
                        }],
                        annotations: [{
                            x: splitDate,
                            y: maxValue,
                            text: 'Historical → Forecast',
                            showarrow: false,
                            xanchor: 'left',
                            yanchor: 'bottom'
                        }],
                        plot_bgcolor: 'white',
                        paper_bgcolor: 'white'
                    }
                };
            });

            setTargetGraphs(processedGraphs);
        }
    }, [results]);

    const handleDownload = async () => {
        if (!results || !results.length) {
            toast.error('No forecast data available to download');
            return;
        }

        setDownloading(true);
        try {
            // Get data from localStorage
            const storedModelSelections = localStorage.getItem('modelSelections');
            const storedTargets = localStorage.getItem('selectedTargets');
            
            if (!storedModelSelections || !storedTargets) {
                throw new Error('Training configuration not found. Please retrain the models.');
            }

            const modelSelections = JSON.parse(storedModelSelections);
            const targets = JSON.parse(storedTargets);
            
            // Verify each target has a model selection
            const availableTargets = Object.keys(results[0]).filter(key => key !== 'date');
            for (const target of availableTargets) {
                if (!modelSelections[target]) {
                    modelSelections[target] = localStorage.getItem(`model_${target}`) || 'ARIMA'; // fallback to ARIMA
                }
            }

            const horizon = parseInt(localStorage.getItem('forecastHorizon')) || 30;
            const timePeriod = localStorage.getItem('timePeriod') || 'day';
            const aggregationMethod = localStorage.getItem('aggregationMethod') || 'mean';

            toast.info(`Starting download in ${selectedFormat.toUpperCase()} format...`);

            const downloadStartTime = Date.now();
            const success = await getForecast(
                availableTargets,
                modelSelections,
                horizon,
                timePeriod,
                aggregationMethod,
                selectedFormat,
                true // forDownload = true
            );

            if (success) {
                // Only show success toast if download took more than 500ms
                if (Date.now() - downloadStartTime > 500) {
                    toast.success(`Successfully downloaded forecast results as ${selectedFormat.toUpperCase()}`);
                }
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            toast.error(`Failed to download forecast: ${error.message || 'Unknown error'}`);
        } finally {
            setDownloading(false);
        }
    };

    if (loading) {
        return (
            <div className="max-w-4xl mx-auto p-6">
                <div className="flex items-center justify-center space-x-4 text-gray-500">
                    <svg className="animate-spin h-8 w-8" viewBox="0 0 24 24">
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <p className="text-gray-500">Loading forecast results...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="max-w-4xl mx-auto p-6">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-red-600">{error}</p>
                </div>
            </div>
        );
    }

    if (!results || !results.length) {
        return (
            <div className="max-w-4xl mx-auto p-6">
                <p className="text-gray-500">No forecast results available. Please complete the training step first.</p>
            </div>
        );
    }

    return (
        <ErrorBoundary>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-7xl mx-auto p-6 space-y-8"
            >
                <div className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-gray-800">Forecast Results</h2>
                            <p className="text-sm text-gray-500 mt-1">
                                Showing historical data and future predictions for each target variable
                            </p>
                        </div>
                        <div className="flex items-center space-x-4">
                            <select
                                value={selectedFormat}
                                onChange={(e) => setSelectedFormat(e.target.value)}
                                className="rounded-lg border border-gray-200 p-2 text-sm"
                            >
                                <option value="json">JSON</option>
                                <option value="csv">CSV</option>
                                <option value="excel">Excel</option>
                            </select>
                            <Button
                                onClick={handleDownload}
                                disabled={downloading}
                                className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white"
                            >
                                {downloading ? (
                                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                ) : (
                                    <FiDownload className="h-4 w-4" />
                                )}
                                <span>Download</span>
                            </Button>
                        </div>
                    </div>

                    <div className="space-y-8">
                        {targetGraphs.map((graph, index) => (
                            <div key={index} className="bg-gray-50 rounded-lg p-4">
                                <Plot
                                    data={graph.traces}
                                    layout={graph.layout}
                                    config={{
                                        responsive: true,
                                        displayModeBar: true,
                                        displaylogo: false,
                                        modeBarButtonsToRemove: [
                                            'lasso2d',
                                            'select2d',
                                            'toggleSpikelines'
                                        ]
                                    }}
                                    className="w-full h-[600px]"
                                />
                            </div>
                        ))}
                    </div>
                </div>
            </motion.div>
        </ErrorBoundary>
    );
};

export default Results;