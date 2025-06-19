import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useWorkflow } from '@/contexts/WorkflowContext';
import { Button } from '@/components/ui/button';
import { FiDownload, FiBarChart } from 'react-icons/fi';
import { getForecast } from '@/services/api';
import { toast } from 'sonner';
import ErrorBoundary from '@/components/ErrorBoundary';
import Plot from 'react-plotly.js';
import * as d3 from 'd3';

const modelTypes = [
    { id: 'Prophet', name: 'Prophet', description: 'Best for time series with strong seasonal effects' },
    { id: 'ARIMA', name: 'ARIMA', description: 'Traditional statistical forecasting' },
    { id: 'LSTM', name: 'LSTM', description: 'Deep learning for complex patterns' },
    { id: 'RandomForest', name: 'Random Forest', description: 'Ensemble method for robust predictions' },
    { id: 'EMA', name: 'EMA', description: 'Simple and effective trend following' },
    { id: 'HoltWinters', name: 'Holt-Winters', description: 'Triple exponential smoothing with seasonality' }
];

const Results = () => {    
    const { results, setResults } = useWorkflow();
    const [selectedFormat, setSelectedFormat] = useState('json');
    const [downloading, setDownloading] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [selectedTarget, setSelectedTarget] = useState(null);
    const [accuracyMetrics, setAccuracyMetrics] = useState(null);    const [configuration, setConfiguration] = useState({
        process: {
            horizon: 30,
            timePeriod: 'day',
            aggregationMethod: 'sum'
        },
        training: {
            selectedTargets: [],
            modelSelections: {}
        }    });

    const availableTargets = [
        { id: 'transformOrdNo', name: 'Orders', description: 'Number of unique orders' },
        { id: 'quantity', name: 'Products', description: 'Total products ordered' },
        { id: 'workers_needed', name: 'Employees', description: 'Required workforce' },
        { id: 'woNumber', name: 'Work Orders', description: 'Number of unique work orders' }
    ];    // Enhanced color palette matching AnimatedChart.jsx design
    const getColorPalette = (targetId) => {
        // All targets now use the same professional color scheme from AnimatedChart.jsx
        return {
            historical: '#4f46e5',    // Indigo-600 - consistent historical data color
            forecast: '#f59e0b',      // Amber-500 - consistent forecast color  
            confidence: 'rgba(245, 158, 11, 0.1)',  // Amber with low opacity
            divider: '#ef4444'        // Red-500 - for transition/today line
        };
    };

    // Effect to load initial data
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
    }, []); // Only run once on mount

    // Effect to set initial selected target
    useEffect(() => {
        if (results && results.length > 0 && !selectedTarget) {
            // Find first available target from the selected targets
            const firstTarget = availableTargets.find(target => 
                configuration.training.selectedTargets.includes(target.id) &&
                results.some(r => r[target.id] !== undefined)
            );
            if (firstTarget) {
                setSelectedTarget(firstTarget);
            }
        }
    }, [results, configuration.training.selectedTargets]);    // Load configuration on mount
    useEffect(() => {
        const loadConfiguration = () => {
            try {
                // Try to load complete configuration first
                const completeConfig = localStorage.getItem('completeConfiguration');
                if (completeConfig) {
                    const parsed = JSON.parse(completeConfig);
                    setConfiguration(parsed);
                    return;
                }

                // Fallback to loading individual items
                const horizon = parseInt(localStorage.getItem('forecastHorizon'));
                const timePeriod = localStorage.getItem('timePeriod');
                const aggregationMethod = localStorage.getItem('aggregationMethod');
                const storedTargets = localStorage.getItem('selectedTargets');
                const storedModels = localStorage.getItem('modelSelections');

                if (!horizon || !timePeriod || !aggregationMethod) {
                    console.error('Missing process configuration');
                    return;
                }

                if (!storedTargets || !storedModels) {
                    console.error('Missing training configuration');
                    return;
                }

                const parsedTargets = JSON.parse(storedTargets);
                const parsedModels = JSON.parse(storedModels);

                setConfiguration({
                    process: {
                        horizon,
                        timePeriod,
                        aggregationMethod
                    },
                    training: {
                        selectedTargets: parsedTargets,
                        modelSelections: parsedModels
                    }
                });

                console.log('Loaded configuration:', {
                    horizon,
                    timePeriod,
                    aggregationMethod,
                    targets: parsedTargets,
                    models: parsedModels
                });
            } catch (error) {
                console.error('Error loading configuration:', error);
            }
        };
        
        loadConfiguration();
    }, []);    const createPlotlyChart = (target, data, timePeriod = 'day') => {
        if (!data || data.length === 0 || !target) {
            return { data: [], layout: {} };
        }

        console.log('Raw data before filtering:', data);
        const historicalData = data.filter(d => d.type === 'historical') || [];
        const forecastData = data.filter(d => d.type === 'forecast') || [];
        console.log('Filtered historical data:', historicalData);
        console.log('Filtered forecast data:', forecastData);

        // Ensure we have data to display
        if (historicalData.length === 0 && forecastData.length === 0) {
            return { data: [], layout: {} };
        }

        // Get color palette for this target
        const colors = getColorPalette(target.id);// Function to get appropriate date format and tick angle based on time period
        const getDateFormatting = (period) => {
            switch (period) {
                case 'month':
                    return {
                        tickformat: '%Y %b',  // 2025 Jan, 2025 Feb, etc.
                        tickmode: 'auto',     // Let Plotly handle tick spacing
                        nticks: 12,           // Suggest max 12 ticks for months
                        tickangle: -45        // 45-degree angle
                    };
                case 'week':
                    return {
                        tickformat: '%Y-%m-%d',  // Weekly with full date
                        tickmode: 'auto',        // Let Plotly handle tick spacing
                        nticks: 8,               // Suggest max 8 ticks for weeks
                        tickangle: -30           // 30-degree angle
                    };
                case 'day':
                default:
                    return {
                        tickformat: '%Y-%m-%d',  // Daily format
                        tickmode: 'auto',        // Let Plotly handle tick spacing
                        nticks: 10,              // Suggest max 10 ticks for days
                        tickangle: -45           // 45-degree angle
                    };
            }
        };

        const dateFormatting = getDateFormatting(timePeriod);

        let traces = [];// Add historical data trace
        if (historicalData.length > 0) {            const formatValue = (value, targetId) => {
                if (targetId === 'transformOrdNo' || targetId === 'woNumber') {
                    return Math.round(value); // Orders and Work Orders must be whole numbers
                } else if (targetId === 'quantity') {
                    return Math.round(value); // Products are typically whole units
                } else if (targetId === 'workers_needed') {
                    return Number(value.toFixed(2)); // Workers can have 2 decimal places for part-time
                }
                return value;
            };

            const getHoverTemplate = (targetId) => {
                if (targetId === 'transformOrdNo' || targetId === 'woNumber' || targetId === 'quantity') {
                    return '%{y:.0f}<extra>Historical</extra>';
                }
                return '%{y:.2f}<extra>Historical</extra>';
            };            traces.push({
                name: 'Historical',
                x: historicalData.map(d => new Date(d.date)),
                y: historicalData.map(d => formatValue(d[target.id], target.id)),
                type: 'scatter',
                mode: 'lines',
                line: { 
                    color: colors.historical,  // Dynamic color based on target
                    width: 3,
                    shape: 'spline',  // Smooth curves for better visual appeal
                    smoothing: 0.3
                },
                hovertemplate: getHoverTemplate(target.id),
                connectgaps: true
            });
        }        // Add forecast data and confidence intervals
        if (forecastData.length > 0) {
            // Get the last historical date for proper transition
            const lastHistoricalDate = new Date(historicalData[historicalData.length - 1].date);
            const lastHistoricalValue = historicalData[historicalData.length - 1][target.id];
            const firstForecastDate = new Date(forecastData[0].date);
            const firstForecastValue = forecastData[0][target.id];            // Add transition marker
            traces.push({
                name: 'Transition Point',
                x: [lastHistoricalDate],
                y: [lastHistoricalValue],
                type: 'scatter',
                mode: 'markers',
                marker: { 
                    color: colors.divider,  // Red-500 - matching AnimatedChart today line
                    size: 5,
                    symbol: 'dot',
                    line: {
                        color: '#b91c1c',  // Darker red border
                        width: 2
                    }
                },
                showlegend: false,
                hovertemplate: 'Transition Point<br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
            });            // Add connecting line between historical and forecast
            traces.push({
                name: 'Connection',
                x: [lastHistoricalDate, firstForecastDate],
                y: [lastHistoricalValue, firstForecastValue],
                type: 'scatter',
                mode: 'lines',
                line: { 
                    color: colors.forecast,  // Use forecast color for smooth transition
                    width: 0.5,
                    dash: 'dot'  // Subtle dotted connection
                },
                showlegend: false,
                hoverinfo: 'skip'
            });            // Add confidence intervals (only for forecast period)
            if (forecastData[0]?.upper && forecastData[0]?.lower) {
                traces.push({
                    name: 'Confidence Interval',
                    x: [...forecastData.map(d => new Date(d.date)), ...forecastData.map(d => new Date(d.date)).reverse()],
                    y: [...forecastData.map(d => d.upper), ...forecastData.map(d => d.lower).reverse()],
                    fill: 'toself',
                    fillcolor: colors.confidence,  // Amber with low opacity
                    line: { 
                        color: 'rgba(245, 158, 11, 0.3)',  // Amber border with opacity
                        width: 0
                    },
                    showlegend: false,
                    hoverinfo: 'skip'
                });
            } else {
                // Fallback to a simple ±5% confidence interval for EMA
                const volatility = 0.05; // 5% variation
                traces.push({
                    name: 'Confidence Interval',
                    x: [...forecastData.map(d => new Date(d.date)), ...forecastData.map(d => new Date(d.date)).reverse()],
                    y: [...forecastData.map(d => d[target.id] * (1 + volatility)), 
                       ...forecastData.map(d => d[target.id] * (1 - volatility)).reverse()],
                    fill: 'toself',
                    fillcolor: colors.confidence,  // Amber with low opacity
                    line: { 
                        color: 'rgba(245, 158, 11, 0.3)',  // Amber border with opacity
                        width: 0
                    },
                    showlegend: false,
                    hoverinfo: 'skip'
                });
            }// Add forecast line (only for future dates)
            traces.push({
                name: 'Forecast',
                x: forecastData.map(d => new Date(d.date)),
                y: forecastData.map(d => target.id === 'transformOrdNo' ? Math.round(d[target.id]) : d[target.id]),
                type: 'scatter',
                mode: 'lines+markers',  // Add markers for forecast points
                line: { 
                    color: colors.forecast,  // Amber-500 - matching AnimatedChart
                    width: 3,
                    shape: 'spline',  // Smooth curves
                    smoothing: 0.3,
                    dash: 'dash'  // Dashed line for forecast - matching AnimatedChart
                },
                marker: {
                    color: colors.forecast,
                    size: 6,
                    symbol: 'circle',
                    line: {
                        color: colors.historical,  // Indigo border for contrast
                        width: 1
                    }
                },
                hovertemplate: target.id === 'transformOrdNo' ? '%{y:.0f}<extra>Forecast</extra>' : '%{y:,.2f}<extra>Forecast</extra>',
                connectgaps: true
            });
        }        // Create layout with enhanced styling and colors
        const layout = {
            autosize: true,
            height: 450,
            title: {
                text: `${target.name} Forecast`,
                font: {
                    size: 18,
                    color: '#1f2937',
                    family: 'Inter, sans-serif'
                },
                x: 0.02
            },            xaxis: { 
                title: {
                    text: 'Date',
                    font: {
                        size: 14,
                        color: '#374151'
                    }
                },
                type: 'date',
                tickformat: dateFormatting.tickformat,
                tickmode: dateFormatting.tickmode,
                nticks: dateFormatting.nticks,
                tickangle: dateFormatting.tickangle,
                showgrid: true,
                gridcolor: 'rgba(156, 163, 175, 0.3)',
                gridwidth: 1,
                rangeslider: {
                    visible: true,
                    thickness: 0.05,
                    bgcolor: 'rgba(243, 244, 246, 0.8)',
                    bordercolor: 'rgba(156, 163, 175, 0.5)',
                    borderwidth: 1
                },
                tickfont: {
                    size: 12,
                    color: '#6b7280'
                }
            },
            yaxis: { 
                title: {
                    text: target.name,
                    font: {
                        size: 14,
                        color: '#374151'
                    }
                },
                showgrid: true,
                gridcolor: 'rgba(156, 163, 175, 0.3)',
                gridwidth: 1,
                zeroline: true,
                zerolinecolor: 'rgba(75, 85, 99, 0.4)',
                zerolinewidth: 2,
                tickfont: {
                    size: 12,
                    color: '#6b7280'
                }
            },
            hovermode: 'x unified',
            plot_bgcolor: 'rgba(255, 255, 255, 0.95)',
            paper_bgcolor: 'rgba(255, 255, 255, 1)',
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                bordercolor: 'rgba(156, 163, 175, 0.5)',
                borderwidth: 1,
                font: {
                    size: 12,
                    color: '#374151'
                }
            },
            margin: {
                l: 60,                r: 30,
                t: 60,
                b: 50
            },            annotations: historicalData.length > 0 ? [{
                x: historicalData[historicalData.length - 1].date,
                y: 0.95,
                yref: 'paper',
                text: 'Forecast Start',
                showarrow: true,
                arrowhead: 3,
                arrowcolor: colors.divider,  // Red-500 for consistency
                arrowsize: 1.5,
                ax: 0,
                ay: -30,
                font: {
                    size: 12,
                    color: colors.divider  // Red-500 for consistency
                },
                bgcolor: 'rgba(239, 68, 68, 0.1)',  // Red background with opacity
                bordercolor: colors.divider,  // Red-500 border
                borderwidth: 1
            }] : []
        };        // Add divider line at the last historical date if we have both types of data
        if (historicalData.length > 0 && forecastData.length > 0) {
            const lastHistoricalDate = new Date(historicalData[historicalData.length - 1].date);
            layout.shapes = [{
                type: 'line',
                x0: lastHistoricalDate,
                x1: lastHistoricalDate,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {
                    color: colors.divider,  // Red-500 to match AnimatedChart today line
                    width: 2,
                    dash: 'dash'
                }
            }];
        }        return { data: traces, layout };
    };    // Effect to update accuracy metrics when target changes
    useEffect(() => {
        if (results && results.length > 0 && selectedTarget) {
            const historicalData = results.filter(d => d.type === 'historical');
            const forecastData = results.filter(d => d.type === 'forecast');
            const metrics = calculateAccuracyMetrics(historicalData, forecastData, selectedTarget.id);
            setAccuracyMetrics(metrics);
        }
    }, [selectedTarget, results]); // Only run when selected target or results change

    // Memoize chart data for all targets to avoid re-computation and fix hooks issue
    const chartsData = useMemo(() => {
        if (!results || !results.length || !configuration.training.selectedTargets) {
            return {};
        }
        
        const data = {};
        availableTargets
            .filter(target => 
                configuration.training.selectedTargets.includes(target.id) && 
                results.some(r => r[target.id] !== undefined)
            )
            .forEach(target => {
                data[target.id] = createPlotlyChart(target, results, configuration.process?.timePeriod || 'day');
            });
        
        return data;
    }, [results, configuration.training.selectedTargets, configuration.process?.timePeriod]);

    const calculateAccuracyMetrics = (historicalData, forecastData, targetId) => {
        if (!historicalData || !forecastData || historicalData.length === 0 || forecastData.length === 0) {
            return null;
        }

        // Get overlapping period data
        const lastHistoricalDate = new Date(historicalData[historicalData.length - 1].date);
        const historicalValues = historicalData.map(d => d[targetId]);
        const forecastValues = forecastData
            .filter(d => new Date(d.date) <= lastHistoricalDate)
            .map(d => d[targetId]);

        if (historicalValues.length === 0 || forecastValues.length === 0) {
            return null;
        }

        // Calculate metrics
        const mse = historicalValues.reduce((sum, actual, i) => {
            const predicted = forecastValues[i] || 0;
            return sum + Math.pow(actual - predicted, 2);
        }, 0) / historicalValues.length;

        const rmse = Math.sqrt(mse);
        
        const mae = historicalValues.reduce((sum, actual, i) => {
            const predicted = forecastValues[i] || 0;
            return sum + Math.abs(actual - predicted);
        }, 0) / historicalValues.length;

        const mape = historicalValues.reduce((sum, actual, i) => {
            const predicted = forecastValues[i] || 0;
            return sum + (Math.abs((actual - predicted) / actual) * 100);
        }, 0) / historicalValues.length;

        return {
            rmse: rmse.toFixed(2),
            mae: mae.toFixed(2),
            mape: mape.toFixed(2) + '%'
        };
    };    const handleDownload = async () => {
        if (!results || !results.length) {
            toast.error('No forecast data available to download');
            return;
        }

        setDownloading(true);
        try {
            if (!configuration.training.modelSelections || !configuration.training.selectedTargets) {
                throw new Error('Training configuration not found. Please retrain the models.');
            }

            if (!configuration.process.horizon || !configuration.process.timePeriod || !configuration.process.aggregationMethod) {
                throw new Error('Process settings not found. Please reconfigure process settings.');
            }

            // Format models object correctly for the API
            const modelSelections = {};
            for (const target of configuration.training.selectedTargets) {
                const modelInfo = configuration.training.modelSelections[target];
                // The backend expects just the model type string (e.g. 'Prophet', 'ARIMA', etc.)
                modelSelections[target] = modelInfo?.id || modelInfo;
            }

            toast.info(`Starting download in ${selectedFormat.toUpperCase()} format...`);

            const downloadStartTime = Date.now();
            const success = await getForecast(
                configuration.training.selectedTargets,
                modelSelections,
                configuration.process.horizon,
                configuration.process.timePeriod,
                configuration.process.aggregationMethod,
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
                    {/* Header and Download Section */}
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-gray-800">Forecast Results</h2>
                            <p className="text-sm text-gray-500 mt-1">
                                Select a target variable to view its forecast details
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

                    {/* Configuration Summary */}
                    <div className="mb-8 p-4 bg-gray-50 rounded-lg">
                        <h3 className="text-lg font-semibold text-gray-800 mb-4">Forecast Configuration</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Process Settings */}
                            <div>
                                <h4 className="font-medium text-gray-700 mb-2">Process Settings</h4>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Time Period:</span>
                                        <span className="font-medium capitalize">{configuration.process.timePeriod}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Aggregation Method:</span>
                                        <span className="font-medium capitalize">{configuration.process.aggregationMethod}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Forecast Horizon:</span>
                                        <span className="font-medium">{configuration.process.horizon} {configuration.process.timePeriod}s</span>
                                    </div>
                                </div>
                            </div>                            {/* Model Selections */}
                            <div>
                                <h4 className="font-medium text-gray-700 mb-2">Selected Models</h4>
                                <div className="space-y-2 text-sm">                                    {availableTargets
                                        .filter(target => configuration.training.selectedTargets.includes(target.id))
                                        .map(target => {
                                            const modelInfo = configuration.training.modelSelections[target.id];
                                            return (
                                                <div key={target.id} className="flex justify-between">
                                                    <span className="text-gray-600">{target.name}:</span>
                                                    <span className="font-medium">
                                                        {modelInfo?.name || 'Not selected'}
                                                    </span>
                                                </div>
                                            );
                                    })}
                                </div>
                            </div>
                        </div>
                    </div>{/* Forecast Visualizations */}
                    <div className="space-y-8">                        {availableTargets
                            .filter(target => 
                                configuration.training.selectedTargets.includes(target.id) && 
                                results.some(r => r[target.id] !== undefined)
                            )
                            .map((target) => {
                                // Get pre-computed chart data
                                const chartData = chartsData[target.id] || { data: [], layout: {} };
                                
                                return (
                            <div key={target.id} className="bg-white rounded-xl shadow-md p-6 border border-gray-100">
                                <h3 className="text-xl font-semibold text-gray-800 mb-4">{target.name} Forecast</h3>
                                <p className="text-sm text-gray-500 mb-4">{target.description}</p>
                                
                                {/* Accuracy Metrics */}
                                {calculateAccuracyMetrics(
                                    results.filter(d => d.type === 'historical'),
                                    results.filter(d => d.type === 'forecast'),
                                    target.id
                                ) && (
                                    <div className="grid grid-cols-3 gap-4 mb-6">
                                        <div className="bg-gray-50 p-4 rounded-lg">
                                            <div className="text-sm text-gray-500">RMSE</div>
                                            <div className="text-xl font-semibold text-gray-800">
                                                {calculateAccuracyMetrics(
                                                    results.filter(d => d.type === 'historical'),
                                                    results.filter(d => d.type === 'forecast'),
                                                    target.id
                                                ).rmse}
                                            </div>
                                        </div>
                                        <div className="bg-gray-50 p-4 rounded-lg">
                                            <div className="text-sm text-gray-500">MAE</div>
                                            <div className="text-xl font-semibold text-gray-800">
                                                {calculateAccuracyMetrics(
                                                    results.filter(d => d.type === 'historical'),
                                                    results.filter(d => d.type === 'forecast'),
                                                    target.id
                                                ).mae}
                                            </div>
                                        </div>
                                        <div className="bg-gray-50 p-4 rounded-lg">
                                            <div className="text-sm text-gray-500">MAPE</div>
                                            <div className="text-xl font-semibold text-gray-800">
                                                {calculateAccuracyMetrics(
                                                    results.filter(d => d.type === 'historical'),
                                                    results.filter(d => d.type === 'forecast'),
                                                    target.id
                                                ).mape}
                                            </div>
                                        </div>
                                    </div>
                                )}
                                  {/* Plotly Chart */}
                                {chartData.data && chartData.data.length > 0 ? (
                                    <Plot
                                        data={chartData.data}
                                        layout={chartData.layout}
                                        style={{ width: '100%' }}
                                        config={{ 
                                            responsive: true,
                                            displayModeBar: true,
                                            modeBarButtonsToRemove: ['lasso2d', 'select2d']
                                        }}
                                    />
                                ) : (
                                    <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
                                        <p className="text-gray-500">No chart data available for {target.name}</p>
                                    </div>
                                )}
                            </div>
                                );
                            })}
                    </div>                    {/* Empty to remove the selected target visualization section */}
                </div>
            </motion.div>
        </ErrorBoundary>
    );
};

export default Results;