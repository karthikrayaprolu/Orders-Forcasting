import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useWorkflow } from '@/contexts/WorkflowContext';
import { Button } from '@/components/ui/button';
import Plot from 'react-plotly.js';
import { FiDownload, FiCheck } from 'react-icons/fi';
import { getForecast } from '@/services/api';
import { toast } from 'sonner';

const Results = () => {
    const { results, setResults } = useWorkflow();
    const [selectedFormat, setSelectedFormat] = useState('json');
    const [downloading, setDownloading] = useState(false);
    const [chartData, setChartData] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        // Load saved configurations and results
        const storedSelections = localStorage.getItem('modelSelections');
        const storedTargets = localStorage.getItem('selectedTargets');
        const horizon = parseInt(localStorage.getItem('forecastHorizon')) || 30;
        const timePeriod = localStorage.getItem('timePeriod') || 'day';
        const aggregationMethod = localStorage.getItem('aggregationMethod') || 'mean';
        
        if (!results && storedSelections && storedTargets) {
            setLoading(true);
            const modelSelections = JSON.parse(storedSelections);
            const targets = JSON.parse(storedTargets);
            
            // Reload forecast data if missing
            getForecast(targets, modelSelections, horizon, timePeriod, aggregationMethod, 'json')
                .then(data => {
                    if (data && !data.error) {
                        setResults(data);
                    } else {
                        throw new Error(data?.error || 'Failed to load forecast data');
                    }
                })
                .catch(error => {
                    console.error('Error reloading forecast:', error);
                    toast.error(`Failed to load forecast: ${error.message}`);
                })
                .finally(() => setLoading(false));
        }
    }, []);

    useEffect(() => {
        if (results && results.length > 0) {
            const targets = Object.keys(results[0]).filter(key => key !== 'date');
            const traces = targets.map(target => ({
                type: 'scatter',
                mode: 'lines+markers',
                name: target.charAt(0).toUpperCase() + target.slice(1),
                x: results.map(r => r.date),
                y: results.map(r => r[target]),
                line: { color: getRandomColor(), width: 2 },
                marker: { size: 6 }
            }));
            setChartData(traces);
        }
    }, [results]);

    const getRandomColor = () => {
        const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444'];
        return colors[Math.floor(Math.random() * colors.length)];
    };

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

            console.log('Download request params:', {
                targets: availableTargets,
                models: modelSelections,
                horizon,
                timePeriod,
                aggregationMethod,
                outputFormat: selectedFormat
            });

            toast.info(`Starting download in ${selectedFormat.toUpperCase()} format...`);

            const downloadStartTime = Date.now();
            const success = await getForecast(
                availableTargets,
                modelSelections,
                horizon,
                timePeriod,
                aggregationMethod,
                selectedFormat
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
            toast.error(`Failed to download results: ${error.message}`);
        } finally {
            setDownloading(false);
        }
    };

    if (loading) {
        return (
            <div className="max-w-4xl mx-auto p-6">
                <p className="text-gray-500">Loading forecast results...</p>
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
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-4xl mx-auto p-6 bg-white rounded-xl shadow-lg space-y-8"
        >
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-800">Forecast Results</h2>
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
                        className="flex items-center space-x-2"
                    >
                        {downloading ? (
                            <>
                                <svg className="animate-spin h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Downloading...
                            </>
                        ) : (
                            <>
                                <FiDownload className="w-4 h-4" />
                                Download
                            </>
                        )}
                    </Button>
                </div>
            </div>

            <div className="space-y-8">
                <Plot
                    data={chartData}
                    layout={{
                        title: 'Forecast Results',
                        autosize: true,
                        height: 500,
                        margin: { t: 60, r: 40, b: 40, l: 60 },
                        showlegend: true,
                        xaxis: {
                            title: 'Date',
                            showgrid: true,
                            gridcolor: '#f3f4f6'
                        },
                        yaxis: {
                            title: 'Value',
                            showgrid: true,
                            gridcolor: '#f3f4f6'
                        },
                        plot_bgcolor: 'white',
                        paper_bgcolor: 'white',
                    }}
                    config={{
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                    }}
                    className="w-full"
                />
            </div>
        </motion.div>
    );
};

export default Results;