import React, { useState } from 'react';
import { getForecast } from '../services/api';

const availableModels = ['ARIMA', 'Prophet', 'LSTM', 'RandomForest'];
const availableTargets = ['orders', 'products', 'employees', 'throughput'];

const ProcessStep = ({ onComplete }) => {
    const [selectedTargets, setSelectedTargets] = useState([]);
    const [modelSelections, setModelSelections] = useState({});
    const [horizon, setHorizon] = useState(30);
    const [outputFormat, setOutputFormat] = useState('json');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleTargetChange = (target) => {
        setSelectedTargets(prev => {
            if (prev.includes(target)) {
                const newTargets = prev.filter(t => t !== target);
                setModelSelections(prev => {
                    const newSelections = { ...prev };
                    delete newSelections[target];
                    return newSelections;
                });
                return newTargets;
            }
            return [...prev, target];
        });
    };

    const handleModelChange = (target, model) => {
        setModelSelections(prev => ({
            ...prev,
            [target]: model
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            if (selectedTargets.length === 0) {
                throw new Error('Please select at least one target');
            }
            if (Object.keys(modelSelections).length !== selectedTargets.length) {
                throw new Error('Please select models for all targets');
            }

            const result = await getForecast(selectedTargets, modelSelections, horizon, outputFormat);
            onComplete && onComplete(result);
        } catch (err) {
            setError(err.message || 'Failed to generate forecast');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-6">
            <h2 className="text-2xl font-bold mb-6">Configure Forecast Parameters</h2>
            
            <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Select Targets</h3>
                    <div className="grid grid-cols-2 gap-4">
                        {availableTargets.map(target => (
                            <div key={target} className="flex items-center">
                                <input
                                    type="checkbox"
                                    id={target}
                                    checked={selectedTargets.includes(target)}
                                    onChange={() => handleTargetChange(target)}
                                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                />
                                <label htmlFor={target} className="ml-2 text-sm text-gray-700 capitalize">
                                    {target}
                                </label>
                            </div>
                        ))}
                    </div>
                </div>

                {selectedTargets.length > 0 && (
                    <div>
                        <h3 className="text-lg font-medium text-gray-900 mb-4">Select Models</h3>
                        <div className="space-y-4">
                            {selectedTargets.map(target => (
                                <div key={target} className="flex items-center space-x-4">
                                    <span className="text-sm text-gray-700 capitalize w-24">{target}:</span>
                                    <select
                                        value={modelSelections[target] || ''}
                                        onChange={(e) => handleModelChange(target, e.target.value)}
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                                    >
                                        <option value="">Select a model</option>
                                        {availableModels.map(model => (
                                            <option key={model} value={model}>{model}</option>
                                        ))}
                                    </select>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Forecast Horizon</h3>
                    <input
                        type="number"
                        min="1"
                        max="365"
                        value={horizon}
                        onChange={(e) => setHorizon(parseInt(e.target.value))}
                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                    />
                    <p className="mt-1 text-sm text-gray-500">Number of days to forecast (1-365)</p>
                </div>

                <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Output Format</h3>
                    <select
                        value={outputFormat}
                        onChange={(e) => setOutputFormat(e.target.value)}
                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                    >
                        <option value="json">JSON</option>
                        <option value="csv">CSV</option>
                        <option value="excel">Excel</option>
                    </select>
                </div>

                {error && (
                    <div className="text-red-600 text-sm mt-2">
                        {error}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading || selectedTargets.length === 0}
                    className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white 
                        ${loading || selectedTargets.length === 0
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
                        }`}
                >
                    {loading ? 'Processing...' : 'Generate Forecast'}
                </button>
            </form>
        </div>
    );
};

export default ProcessStep;