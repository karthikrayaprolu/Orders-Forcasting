import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiClock, FiAlertCircle, FiX, FiCheck, FiBarChart2 } from 'react-icons/fi';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { useWorkflow } from '@/contexts/WorkflowContext';

const ProcessStep = () => {    const { completeStep, STEPS, process, setProcess } = useWorkflow();
    const [horizon, setHorizon] = useState('');
    const [timePeriod, setTimePeriod] = useState(process?.timePeriod || 'day');
    const [aggregationMethod, setAggregationMethod] = useState(process?.aggregationMethod || 'sum');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');    const timePeriods = [        { 
            id: 'day', 
            label: 'Days', 
            description: 'Daily aggregation, calculated at midnight. For multiple orders in a day, values are combined using the selected method (e.g., summed, averaged).',
            details: 'Best for short-term, detailed forecasting and daily capacity planning'
        },        { 
            id: 'week', 
            label: 'Weeks', 
            description: 'Weekly aggregation, calculated every Saturday. All orders Monday through Saturday are combined using the selected method.',
            details: 'Good for medium-term trends, weekly planning, and seasonal patterns'
        },        { 
            id: 'month', 
            label: 'Months', 
            description: 'Monthly aggregation, calculated at month end. All orders within the month are combined using the selected method.',
            details: 'Best for long-term strategic planning and monthly capacity assessment'
        }
    ];    const aggregationMethods = [        { 
            id: 'mean', 
            label: 'Average', 
            description: 'Calculate average values for multiple orders in each time period. Use only if you need to analyze typical values rather than totals.',
            effect: 'Shows average values but may not reflect actual volumes' 
        },{ 
            id: 'sum', 
            label: 'Sum (Recommended)', 
            description: 'Add up all values within each time period. Default method - best for total volume, orders, and capacity forecasting.',
            effect: 'Shows actual totals and preserves real quantities in forecasts' 
        },{ 
            id: 'min', 
            label: 'Minimum', 
            description: 'Use smallest order size in each time period. Good for minimum capacity planning.',
            effect: 'Shows baseline order levels and minimum required capacity'
        },        { 
            id: 'max', 
            label: 'Maximum', 
            description: 'Use largest order size in each time period. Essential for peak capacity planning.',
            effect: 'Shows peak demand levels and maximum capacity needs'
        }
    ];

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            if (!horizon || horizon <= 0) {
                throw new Error('Please enter a valid forecast period');
            }
            
            setProcess({
                horizon: parseInt(horizon),
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
    };    const handleHorizonChange = (value) => {
        const maxValue = timePeriod === 'day' ? 365 : timePeriod === 'week' ? 52 : 12;

        // Handle empty input
        if (!value && value !== 0) {
            setHorizon('');
            return;
        }

        // Convert to number
        let numValue = parseInt(value);
        
        // Handle invalid number
        if (isNaN(numValue)) {
            setHorizon('');
            return;
        }

        // Clamp the value between 1 and maxValue
        numValue = Math.max(1, Math.min(numValue, maxValue));
        setHorizon(numValue);
        
        // Show warning if original input exceeded max
        if (parseInt(value) > maxValue) {
            toast.error(`Maximum forecast period is ${maxValue} ${timePeriod}s (1 year)`);
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
                                        onClick={() => {                                            if (period.id !== timePeriod) {
                                                setTimePeriod(period.id);
                                                setHorizon('');
                                                toast.info(`Please enter the forecast period in ${period.label.toLowerCase()}`);
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
                        <div className="flex items-center space-x-4">                            <input
                                type="range"
                                min="1"
                                max={timePeriod === 'day' ? 365 : timePeriod === 'week' ? 52 : 12}
                                value={horizon || 1}
                                onChange={(e) => handleHorizonChange(e.target.value)}
                                className={`flex-1 h-2 rounded-lg appearance-none cursor-pointer accent-blue-600 ${
                                    horizon ? 'bg-blue-200' : 'bg-gray-200'
                                }`}
                            />
                            <div className="flex items-center space-x-2">
                                <input
                                    type="number"
                                    placeholder="Enter value"
                                    value={horizon}
                                    onChange={(e) => handleHorizonChange(e.target.value)}
                                    className={`w-24 text-center rounded-lg border py-1.5 text-sm focus:ring-1 focus:ring-blue-500 ${
                                        !horizon ? 'border-gray-200' :
                                        parseInt(horizon) > (timePeriod === 'day' ? 365 : timePeriod === 'week' ? 52 : 12) 
                                            ? 'border-red-500 focus:border-red-500' 
                                            : 'border-green-500 focus:border-green-500'
                                    }`}
                                />
                                <span className="text-sm font-medium text-gray-600">{timePeriod}s</span>
                            </div>
                        </div>                        <div className="flex justify-between text-xs text-gray-500 px-1">
                            <span>Min: 1 {timePeriod}</span>
                            <span>Max: {timePeriod === 'day' ? 365 : timePeriod === 'week' ? 52 : 12} {timePeriod}s (1 year)</span>
                            <span>Current: {horizon ? `${horizon} ${timePeriod}s` : 'Not set'}</span>
                        </div>
                        {horizon !== '' && parseInt(horizon) > (timePeriod === 'day' ? 365 : timePeriod === 'week' ? 52 : 12) && (
                            <div className="text-red-500 text-xs">
                                Warning: Value exceeds maximum limit of {timePeriod === 'day' ? 365 : timePeriod === 'week' ? 52 : 12} {timePeriod}s
                            </div>
                        )}
                    </div>                <div className="space-y-4">
                        <div className="p-4 bg-blue-50 border border-blue-100 rounded-lg">
                            <p className="text-sm text-blue-700 mb-2">
                                <strong>Guidance:</strong> {timePeriods.find(p => p.id === timePeriod)?.description} Adjust based on your needs - our models will adapt to provide the best possible predictions.
                            </p>
                            <div className="mt-3 text-sm text-blue-600">
                                <strong>Multiple Orders Handling:</strong>
                                <ul className="list-disc list-inside mt-1 space-y-1">
                                    <li>Orders within the same {timePeriod} are combined using the selected aggregation method</li>
                                    <li>Each {timePeriod}'s forecast represents the {aggregationMethod === 'mean' ? 'average' : 
                                        aggregationMethod === 'sum' ? 'total' : 
                                        aggregationMethod === 'min' ? 'minimum' : 'maximum'} value expected</li>
                                    <li>Perfect for {aggregationMethod === 'mean' ? 'typical demand patterns' : 
                                        aggregationMethod === 'sum' ? 'total volume planning' : 
                                        aggregationMethod === 'min' ? 'baseline capacity' : 'peak capacity planning'}</li>
                                </ul>
                            </div>
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



