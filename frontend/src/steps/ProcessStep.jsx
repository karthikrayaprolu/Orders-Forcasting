import React, { useState } from 'react';
import { getForecast } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';
import { FiCheck, FiChevronDown, FiAlertCircle, FiClock, FiDownload } from 'react-icons/fi';

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
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="max-w-4xl mx-auto p-8 bg-white rounded-xl shadow-lg border border-gray-100"
        >
            <div className="mb-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Configure Forecast Parameters</h2>
                <p className="text-gray-600">Select your forecasting targets and models to generate predictions</p>
            </div>
            
            <form onSubmit={handleSubmit} className="space-y-8">
                <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                        <span className="bg-blue-100 text-blue-600 p-1.5 rounded-lg mr-3">
                            <FiCheck className="h-5 w-5" />
                        </span>
                        Select Targets
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {availableTargets.map(target => (
                            <motion.div 
                                key={target}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                className="flex items-center"
                            >
                                <input
                                    type="checkbox"
                                    id={target}
                                    checked={selectedTargets.includes(target)}
                                    onChange={() => handleTargetChange(target)}
                                    className="hidden"
                                />
                                <label 
                                    htmlFor={target} 
                                    className={`flex items-center w-full p-3 rounded-lg border-2 cursor-pointer transition-all ${
                                        selectedTargets.includes(target)
                                            ? 'border-blue-500 bg-blue-50'
                                            : 'border-gray-200 hover:border-gray-300'
                                    }`}
                                >
                                    <div className={`flex items-center justify-center h-5 w-5 rounded border mr-3 ${
                                        selectedTargets.includes(target)
                                            ? 'bg-blue-500 border-blue-500 text-white'
                                            : 'bg-white border-gray-300'
                                    }`}>
                                        {selectedTargets.includes(target) && (
                                            <FiCheck className="h-3 w-3" />
                                        )}
                                    </div>
                                    <span className="text-sm font-medium text-gray-700 capitalize">
                                        {target}
                                    </span>
                                </label>
                            </motion.div>
                        ))}
                    </div>
                </div>

                <AnimatePresence>
                    {selectedTargets.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.3 }}
                            className="space-y-4 overflow-hidden"
                        >
                            <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                                <span className="bg-purple-100 text-purple-600 p-1.5 rounded-lg mr-3">
                                    <FiChevronDown className="h-5 w-5" />
                                </span>
                                Select Models
                            </h3>
                            <div className="space-y-3">
                                {selectedTargets.map(target => (
                                    <div key={target} className="flex flex-col sm:flex-row sm:items-center gap-3">
                                        <span className="text-sm font-medium text-gray-700 capitalize sm:w-32">
                                            {target}:
                                        </span>
                                        <select
                                            value={modelSelections[target] || ''}
                                            onChange={(e) => handleModelChange(target, e.target.value)}
                                            className="flex-1 block w-full pl-4 pr-10 py-2.5 text-base border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 sm:text-sm rounded-lg bg-white shadow-sm"
                                        >
                                            <option value="">Select a model</option>
                                            {availableModels.map(model => (
                                                <option key={model} value={model}>{model}</option>
                                            ))}
                                        </select>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                        <span className="bg-amber-100 text-amber-600 p-1.5 rounded-lg mr-3">
                            <FiClock className="h-5 w-5" />
                        </span>
                        Forecast Horizon
                    </h3>
                    <div className="flex items-center space-x-4">
                        <input
                            type="range"
                            min="1"
                            max="365"
                            value={horizon}
                            onChange={(e) => setHorizon(parseInt(e.target.value))}
                            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        />
                        <span className="text-sm font-medium bg-gray-100 px-3 py-1.5 rounded-lg w-16 text-center">
                            {horizon} days
                        </span>
                    </div>
                    <p className="text-xs text-gray-500">Number of days to forecast (1-365)</p>
                </div>

                <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                        <span className="bg-green-100 text-green-600 p-1.5 rounded-lg mr-3">
                            <FiDownload className="h-5 w-5" />
                        </span>
                        Output Format
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        {['json', 'csv', 'excel'].map(format => (
                            <motion.div 
                                key={format}
                                whileHover={{ scale: 1.02 }}
                                className="flex items-center"
                            >
                                <input
                                    type="radio"
                                    id={format}
                                    name="outputFormat"
                                    value={format}
                                    checked={outputFormat === format}
                                    onChange={() => setOutputFormat(format)}
                                    className="hidden"
                                />
                                <label 
                                    htmlFor={format} 
                                    className={`flex items-center justify-center w-full p-3 rounded-lg border-2 cursor-pointer transition-all ${
                                        outputFormat === format
                                            ? 'border-green-500 bg-green-50'
                                            : 'border-gray-200 hover:border-gray-300'
                                    }`}
                                >
                                    <div className={`flex items-center justify-center h-5 w-5 rounded-full border mr-3 ${
                                        outputFormat === format
                                            ? 'bg-green-500 border-green-500'
                                            : 'bg-white border-gray-300'
                                    }`}>
                                        {outputFormat === format && (
                                            <div className="h-2 w-2 rounded-full bg-white"></div>
                                        )}
                                    </div>
                                    <span className="text-sm font-medium text-gray-700 uppercase">
                                        {format}
                                    </span>
                                </label>
                            </motion.div>
                        ))}
                    </div>
                </div>

                <AnimatePresence>
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.3 }}
                            className="overflow-hidden"
                        >
                            <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm flex items-start">
                                <FiAlertCircle className="flex-shrink-0 mt-0.5 mr-2 text-red-500" />
                                <span>{error}</span>
                                <button 
                                    onClick={() => setError('')}
                                    className="ml-auto p-1 rounded-full hover:bg-red-100 transition-colors"
                                >
                                    <FiX className="h-4 w-4" />
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="pt-2">
                    <button
                        type="submit"
                        disabled={loading || selectedTargets.length === 0}
                        className={`w-full flex justify-center items-center py-3 px-6 rounded-xl shadow-md text-sm font-medium text-white transition-all duration-300 ${
                            loading || selectedTargets.length === 0
                                ? 'bg-gray-400 cursor-not-allowed'
                                : 'bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 hover:shadow-lg'
                        }`}
                    >
                        {loading ? (
                            <div className="flex items-center space-x-2">
                                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                <span>Processing...</span>
                            </div>
                        ) : (
                            <span className="flex items-center space-x-2">
                                <FiCheck className="h-5 w-5" />
                                <span>Generate Forecast</span>
                            </span>
                        )}
                    </button>
                </div>
            </form>
        </motion.div>
    );
};

export default ProcessStep;