import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiClock, FiAlertCircle, FiX, FiCheck, FiBarChart2 } from 'react-icons/fi';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { useWorkflow } from '@/contexts/WorkflowContext';

const ProcessStep = () => {
    const { completeStep, STEPS, process, setProcess } = useWorkflow();
    const [horizon, setHorizon] = useState(process?.horizon || 30);
    const [timePeriod, setTimePeriod] = useState(process?.timePeriod || 'day');
    const [aggregationMethod, setAggregationMethod] = useState(process?.aggregationMethod || 'mean');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const timePeriods = [
        { 
            id: 'day', 
            label: 'Days', 
            description: 'Best for short-term operational planning and detailed patterns. Recommended for 1-90 day forecasts.'
        },
        { 
            id: 'week', 
            label: 'Weeks', 
            description: 'Ideal for medium-term planning and reducing daily noise. Best for 1-52 week forecasts.'
        },
        { 
            id: 'month', 
            label: 'Months', 
            description: 'Perfect for long-term strategic planning and seasonal patterns. Optimal for 1-24 month forecasts.'
        }
    ];

    const aggregationMethods = [
        { id: 'mean', label: 'Average', description: 'Mean value over the period' },
        { id: 'sum', label: 'Sum', description: 'Total sum over the period' },
        { id: 'min', label: 'Minimum', description: 'Lowest value in the period' },
        { id: 'max', label: 'Maximum', description: 'Highest value in the period' }
    ];

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            if (horizon <= 0) {
                throw new Error('Horizon must be a positive number');
            }
            
            setProcess({
                horizon,
                timePeriod,
                aggregationMethod
            });

            localStorage.setItem('forecastHorizon', horizon.toString());
            localStorage.setItem('timePeriod', timePeriod);
            localStorage.setItem('aggregationMethod', aggregationMethod);

            toast.success('Process configuration saved');
            completeStep(STEPS.PROCESS);
        } catch (err) {
            setError(err.message || 'Failed to save process configuration');
            toast.error(err.message || 'Failed to save process configuration');
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
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Configure Forecast Settings</h2>
                <p className="text-gray-600">Choose your forecast period and aggregation preferences</p>
            </div>
            
            <form onSubmit={handleSubmit} className="space-y-8">
                {/* Time Period Selection */}
                <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                        <FiClock className="mr-2 h-5 w-5 text-blue-500" />
                        Time Period
                    </h3>
                    <div className="space-y-4">
                        <div className="flex justify-center mb-4">
                            <div className="inline-flex rounded-lg border border-gray-200 p-1">
                                {timePeriods.map(period => (
                                    <button
                                        key={period.id}
                                        type="button"
                                        onClick={() => {
                                            if (period.id !== timePeriod) {
                                                setTimePeriod(period.id);
                                                const defaults = { day: 30, week: 12, month: 6 };
                                                setHorizon(defaults[period.id]);
                                                toast.info(`Forecast horizon set to ${defaults[period.id]} ${period.label.toLowerCase()} by default`);
                                            }
                                        }}
                                        className={`px-6 py-2.5 rounded-md transition-all flex items-center space-x-2 ${
                                            timePeriod === period.id
                                                ? 'bg-blue-500 text-white font-medium shadow-sm'
                                                : 'hover:bg-gray-100 text-gray-700'
                                        }`}
                                    >
                                        <span>{period.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                        {timePeriod && (
                            <div className="text-center">
                                <span className="text-sm text-gray-600">
                                    {timePeriods.find(p => p.id === timePeriod)?.description}
                                </span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Aggregation Method Selection */}
                {timePeriod !== 'day' && (
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                            <FiBarChart2 className="mr-2 h-5 w-5 text-blue-500" />
                            Aggregation Method
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {aggregationMethods.map(method => (
                                <button
                                    key={method.id}
                                    type="button"
                                    onClick={() => setAggregationMethod(method.id)}
                                    className={`p-4 rounded-lg border transition-all ${
                                        aggregationMethod === method.id
                                            ? 'border-blue-500 bg-blue-50 text-blue-700'
                                            : 'border-gray-200 hover:border-blue-200 hover:bg-gray-50'
                                    }`}
                                >
                                    <div className="font-semibold">{method.label}</div>
                                    <div className="text-sm text-gray-500">{method.description}</div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Horizon Selection */}
                <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                        <span className="bg-amber-100 text-amber-600 p-1.5 rounded-lg mr-3">
                            <FiClock className="h-5 w-5" />
                        </span>
                        Forecast Horizon
                    </h3>
                    <div className="flex flex-col space-y-4">
                        <div className="flex items-center space-x-4">
                            <input
                                type="range"
                                min="1"
                                max="10000"
                                value={horizon <= 10000 ? horizon : 10000}
                                onChange={(e) => setHorizon(parseInt(e.target.value))}
                                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                            />
                            <div className="flex items-center space-x-2">
                                <input
                                    type="number"
                                    min="1"
                                    value={horizon}
                                    onChange={(e) => {
                                        const value = parseInt(e.target.value);
                                        if (value >= 1) {
                                            setHorizon(value);
                                            const modelRecommendations = {
                                                day: {
                                                    threshold: 90,
                                                    message: 'For forecasts beyond 90 days, consider using weekly or monthly periods for better accuracy'
                                                },
                                                week: {
                                                    threshold: 52,
                                                    message: 'For forecasts beyond 52 weeks, monthly periods often provide more reliable long-term predictions'
                                                },
                                                month: {
                                                    threshold: 24,
                                                    message: 'For forecasts beyond 24 months, our models will adapt to provide the best possible long-term predictions'
                                                }
                                            };

                                            const recommendation = modelRecommendations[timePeriod];
                                            if (recommendation && value > recommendation.threshold) {
                                                toast.info(recommendation.message, {
                                                    duration: 5000,
                                                });
                                            }
                                        }
                                    }}
                                    className="w-24 text-center rounded-lg border border-gray-200 py-1.5 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                />
                                <span className="text-sm font-medium text-gray-600">{timePeriod}s</span>
                            </div>
                        </div>
                        <div className="flex justify-between text-xs text-gray-500 px-1">
                            <span>Min: 1 {timePeriod}</span>
                            <span>Set any value using direct input</span>
                            <span>Current: {horizon} {timePeriod}s</span>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <p className="text-sm text-gray-600">
                            Select how far into the future you want to predict. The slider supports up to 10,000 {timePeriod}s, but you can enter any positive number in the input field.
                        </p>
                        <div className="p-3 bg-blue-50 border border-blue-100 rounded-lg">
                            <p className="text-sm text-blue-700">
                                <strong>Guidance:</strong> {timePeriods.find(p => p.id === timePeriod)?.description} Adjust based on your needs - our models will adapt to provide the best possible predictions.
                            </p>
                        </div>
                    </div>
                </div>

                {error && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm"
                    >
                        <div className="flex items-start">
                            <FiAlertCircle className="flex-shrink-0 mt-0.5 mr-2" />
                            <span>{error}</span>
                            <button 
                                onClick={() => setError('')}
                                className="ml-auto p-1 hover:bg-red-100 rounded-full"
                            >
                                <FiX className="h-4 w-4" />
                            </button>
                        </div>
                    </motion.div>
                )}

                <div className="flex justify-end pt-6">
                    <Button
                        type="submit"
                        disabled={loading}
                        className="w-full sm:w-auto"
                    >
                        {loading ? (
                            <div className="flex items-center space-x-2">
                                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                </svg>
                                <span>Processing...</span>
                            </div>
                        ) : (
                            <div className="flex items-center space-x-2">
                                <FiCheck className="h-5 w-5" />
                                <span>Continue</span>
                            </div>
                        )}
                    </Button>
                </div>
            </form>
        </motion.div>
    );
};

export default ProcessStep;