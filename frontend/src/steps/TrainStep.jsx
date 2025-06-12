import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useWorkflow } from '@/contexts/WorkflowContext';
import { Button } from '../components/ui/button';
import { toast } from 'sonner';
import { Brain, Shuffle, GitMerge, RefreshCw, Target, Check } from 'lucide-react';
import { getForecast } from '@/services/api';

const TrainStep = () => {
    const { completeStep, STEPS, model, setModel, setResults, process } = useWorkflow();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [selectedTargets, setSelectedTargets] = useState([]);
    const [modelSelections, setModelSelections] = useState({});    const availableTargets = [
        { id: 'transformOrdNo', name: 'Orders', description: 'Number of unique orders' },
        { id: 'quantity', name: 'Products', description: 'Total products ordered' },
        { id: 'workers_needed', name: 'Employees', description: 'Required workforce' },
        { id: 'woNumber', name: 'Work Orders', description: 'Number of unique work orders' }
    ];    const modelTypes = [
        { 
            id: 'Prophet', 
            name: 'Prophet', 
            description: 'Best for seasonal patterns and trending data',
            strengths: 'Handles holidays, missing data, and trend changes'
        },
        { 
            id: 'ARIMA', 
            name: 'ARIMA', 
            description: 'Excellent for stable, consistent patterns',
            strengths: 'Good for short-term forecasts with clear trends'
        },
        { 
            id: 'LSTM', 
            name: 'LSTM', 
            description: 'Advanced deep learning for complex patterns',
            strengths: 'Best for long sequences and intricate relationships'
        },
        { 
            id: 'RandomForest', 
            name: 'Random Forest', 
            description: 'Robust ensemble learning method',
            strengths: 'Handles non-linear relationships and outliers well'
        },
        {
            id: 'EMA',
            name: 'EMA',
            description: 'Exponential Moving Average for trend following',
            strengths: 'Simple, fast, and effective for stable trends'
        },
        {
            id: 'HoltWinters',
            name: 'Holt-Winters',
            description: 'Triple exponential smoothing with seasonality',
            strengths: 'Excellent for data with clear seasonal patterns'
        }
    ];const handleTargetChange = (targetId) => {
        setSelectedTargets(prev => {
            if (prev.includes(targetId)) {
                const newTargets = prev.filter(t => t !== targetId);
                const newModelSelections = { ...modelSelections };
                delete newModelSelections[targetId];
                setModelSelections(newModelSelections);
                return newTargets;            } else {
                return [...prev, targetId];
            }
        });
    };

    const handleModelSelect = (targetId, modelId) => {
        setModelSelections(prev => ({
            ...prev,
            [targetId]: modelId
        }));
    };    const validateModelSelections = () => {
        // Only check if models are selected for all targets
        const missingModels = selectedTargets.filter(targetId => !modelSelections[targetId]);
        if (missingModels.length > 0) {
            const targetNames = missingModels
                .map(id => availableTargets.find(t => t.id === id)?.name)
                .filter(Boolean);
            throw new Error(`Please select models for: ${targetNames.join(', ')}`);
        }
    };

    const handleSubmit = async () => {
        setLoading(true);
        setError('');
        try {
            if (!process || !process.horizon || !process.timePeriod || !process.aggregationMethod) {
                throw new Error('Process configuration is incomplete. Please go back to the Process step.');
            }
            
            // Validate model selections
            validateModelSelections();

            // Store selections in localStorage with full model information
            const modelSelectionsWithInfo = {};
            for (const targetId in modelSelections) {
                const modelId = modelSelections[targetId];
                const modelInfo = modelTypes.find(m => m.id === modelId);
                modelSelectionsWithInfo[targetId] = {
                    id: modelId,
                    name: modelInfo?.name || modelId,
                    description: modelInfo?.description || ''
                };
            }
            
            // Store all configurations
            localStorage.setItem('modelSelections', JSON.stringify(modelSelectionsWithInfo));
            localStorage.setItem('selectedTargets', JSON.stringify(selectedTargets));
            localStorage.setItem('forecastHorizon', process.horizon.toString());
            localStorage.setItem('timePeriod', process.timePeriod);
            localStorage.setItem('aggregationMethod', process.aggregationMethod);
            
            // Store the complete configuration state
            const completeConfig = {
                process: {
                    horizon: process.horizon,
                    timePeriod: process.timePeriod,
                    aggregationMethod: process.aggregationMethod
                },
                training: {
                    selectedTargets,
                    modelSelections: modelSelectionsWithInfo
                }
            };
            localStorage.setItem('completeConfiguration', JSON.stringify(completeConfig));

                // Prepare model selections for API request
            const apiModelSelections = {};
            for (const targetId in modelSelections) {
                apiModelSelections[targetId] = modelSelections[targetId];
            }

            // Make the forecast request with process settings
            const forecastData = await getForecast(
                selectedTargets,
                apiModelSelections,
                process.horizon,
                process.timePeriod,
                process.aggregationMethod,
                'json',
                false
            );

            if (forecastData && !forecastData.error) {
                // Store the results
                setResults(forecastData);
                localStorage.setItem('forecastResults', JSON.stringify(forecastData));
                toast.success('Models configured successfully!');
                // Move to next step
                completeStep(STEPS.TRAIN);
            } else {
                throw new Error(forecastData?.error || 'Failed to get forecast data');
            }
        } catch (error) {
            console.error('Training error:', error);
            setError(error.message || 'Failed to train models');
            toast.error(error.message || 'Failed to train models');
        } finally {
            setLoading(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="max-w-4xl mx-auto p-6 bg-white rounded-xl shadow-md space-y-8"
        >
            {/* Header */}
            <div className="flex items-center space-x-4">
                <div className="p-3 bg-blue-100 rounded-full">
                    <Brain className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                    <h2 className="text-2xl font-bold text-gray-800">Model Training Configuration</h2>
                    <p className="text-gray-500">Select targets and assign models for forecasting</p>
                </div>
            </div>

            {/* Target Selection */}
            <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                    <Target className="w-5 h-5 mr-2 text-blue-500" />
                    Select Target Variables
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {availableTargets.map(target => (                        <div
                            key={target.id}
                            onClick={() => handleTargetChange(target.id)}
                            className={`p-4 rounded-lg border-2 cursor-pointer transition-all
                                ${selectedTargets.includes(target.id)
                                    ? 'border-blue-500 bg-blue-50'
                                    : 'border-gray-200 hover:border-gray-300'}`}
                        >
                            <div className="flex items-center justify-between">
                                <h4 className="font-medium text-gray-800">{target.name}</h4>
                                {selectedTargets.includes(target.id) && (
                                    <Check className="w-5 h-5 text-blue-500" />
                                )}
                            </div>
                            <p className="text-sm text-gray-500 mt-1">{target.description}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Model Selection per Target */}
            {selectedTargets.length > 0 && (
                <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                        <Brain className="w-5 h-5 mr-2 text-blue-500" />
                        Select Models for Each Target
                    </h3>
                    {selectedTargets.map(targetId => {
                        const target = availableTargets.find(t => t.id === targetId);
                        return (
                            <div key={targetId} className="space-y-3">
                                <h4 className="font-medium text-gray-700">{target.name}</h4>                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">                                {modelTypes.map(model => {
                                        return (
                                            <div
                                                key={model.id}
                                                onClick={() => handleModelSelect(targetId, model.id)}
                                                className={`p-3 rounded-lg border cursor-pointer transition-all relative
                                                    ${modelSelections[targetId] === model.id
                                                        ? 'border-green-500 bg-green-50'
                                                        : 'border-gray-200 hover:border-gray-300'}`}
                                            >
                                                <div className="flex items-center justify-between">
                                                    <span className="font-medium">{model.name}</span>
                                                    {modelSelections[targetId] === model.id && (
                                                        <Check className="w-4 h-4 text-green-500" />
                                                    )}
                                                </div>
                                                <p className="text-xs text-gray-500 mt-1">{model.description}</p>
                                                <p className="text-xs text-blue-600 mt-1">{model.strengths}</p>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Error Display */}
            {error && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600"
                >
                    {error}
                </motion.div>
            )}

            {/* Submit Button */}
            <div className="flex justify-end pt-4">
                <Button
                    onClick={handleSubmit}
                    disabled={loading || selectedTargets.length === 0}
                    className="w-full md:w-auto"
                >
                    {loading ? (
                        <div className="flex items-center">
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Configuring Models...
                        </div>
                    ) : (
                        'Configure & Continue'
                    )}
                </Button>
            </div>
        </motion.div>
    );
};

export default TrainStep;