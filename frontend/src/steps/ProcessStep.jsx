import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiClock, FiAlertCircle, FiX, FiCheck, FiBarChart2 } from 'react-icons/fi';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { useWorkflow } from '@/contexts/WorkflowContext';

const ProcessStep = () => {
    const { completeStep, STEPS, process, setProcess } = useWorkflow();
    const [horizon, setHorizon] = useState(30); // Default 30 days
    const [timePeriod, setTimePeriod] = useState('day'); // 'day', 'week', 'month'
    const [aggregationMethod, setAggregationMethod] = useState('mean'); // 'mean', 'sum', 'min', 'max'
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const timePeriods = [
        { id: 'day', label: 'Days', description: 'Daily forecasts' },
        { id: 'week', label: 'Weeks', description: 'Weekly aggregated forecasts' },
        { id: 'month', label: 'Months', description: 'Monthly aggregated forecasts' }
    ];

    const aggregationMethods = [
        { id: 'mean', label: 'Average', description: 'Mean value over the period' },
        { id: 'sum', label: 'Sum', description: 'Total sum over the period' },
        { id: 'min', label: 'Minimum', description: 'Lowest value in the period' },
        { id: 'max', label: 'Maximum', description: 'Highest value in the period' }
    ];    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            setProcess(prev => ({
                ...prev,
                horizon,
                timePeriod,
                aggregationMethod
            }));
            await new Promise(resolve => setTimeout(resolve, 800)); // Simulating processing
            toast.success('Process configuration completed!');
            completeStep(STEPS.PROCESS);
            toast.success('Moving to results visualization...');
        } catch (err) {
            setError('Failed to process data. Please try again.');
            toast.error('Failed to process data. Please try again.');
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
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {timePeriods.map(period => (
                            <button
                                key={period.id}
                                type="button"
                                onClick={() => setTimePeriod(period.id)}
                                className={`p-4 rounded-lg border transition-all ${
                                    timePeriod === period.id
                                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                                        : 'border-gray-200 hover:border-blue-200 hover:bg-gray-50'
                                }`}
                            >
                                <div className="font-semibold">{period.label}</div>
                                <div className="text-sm text-gray-500">{period.description}</div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Aggregation Method Selection (for week/month) */}
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
                                max={timePeriod === 'day' ? 365 : timePeriod === 'week' ? 52 : 12}
                                value={horizon}
                                onChange={(e) => setHorizon(parseInt(e.target.value))}
                                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                            />
                            <span className="text-sm font-medium bg-gray-100 px-3 py-1.5 rounded-lg w-24 text-center">
                                {horizon} {timePeriod}s
                            </span>
                        </div>
                        <div className="flex justify-between text-xs text-gray-500 px-1">
                            <span>1 {timePeriod}</span>
                            <span>{timePeriod === 'day' ? '6 months' : timePeriod === 'week' ? '26 weeks' : '6 months'}</span>
                            <span>{timePeriod === 'day' ? '1 year' : timePeriod === 'week' ? '1 year' : '1 year'}</span>
                        </div>
                    </div>
                    <p className="text-sm text-gray-600">
                        Select how far into the future you want to predict. Longer horizons may have increased uncertainty.
                    </p>
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