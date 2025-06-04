import React from 'react';
import { motion } from 'framer-motion';
import { 
    Database, 
    GitBranch, 
    Brain,
    BarChart4,
    CheckCircle2
} from 'lucide-react';
import { useWorkflow } from '@/contexts/WorkflowContext';

const WorkflowHeader = () => {
    const { currentStep, setCurrentStep, completedSteps, canAccessStep, STEPS } = useWorkflow();

    const steps = [
        { id: STEPS.DATABASE, icon: Database, label: 'Upload Data' },
        { id: STEPS.PROCESS, icon: GitBranch, label: 'Process' },
        { id: STEPS.TRAIN, icon: Brain, label: 'Train Models' },
        { id: STEPS.RESULTS, icon: BarChart4, label: 'Results' }
    ];

    const getStepStatus = (stepId) => {
        if (Array.from(completedSteps).includes(stepId)) {
            return 'completed';
        }
        if (currentStep === stepId) {
            return 'current';
        }
        return 'pending';
    };

    const handleStepClick = (stepId) => {
        if (canAccessStep(stepId)) {
            setCurrentStep(stepId);
        }
    };

    return (
        <div className="bg-white shadow-md rounded-lg mx-4 mt-4 p-6">
            <div className="flex items-center justify-between max-w-4xl mx-auto relative">
                {/* Progress Line */}
                <div className="absolute top-1/2 left-0 w-full h-0.5 bg-gray-200 -z-10" />
                <motion.div 
                    className="absolute top-1/2 left-0 h-0.5 bg-green-500 -z-10"
                    initial={{ width: '0%' }}
                    animate={{ 
                        width: `${(steps.findIndex(s => s.id === currentStep) / (steps.length - 1)) * 100}%` 
                    }}
                    transition={{ duration: 0.5 }}
                />

                {/* Steps */}
                {steps.map((step, index) => {
                    const status = getStepStatus(step.id);
                    const Icon = status === 'completed' ? CheckCircle2 : step.icon;
                    const isDisabled = !canAccessStep(step.id);

                    return (
                        <div key={step.id} className="flex flex-col items-center">
                            <motion.div
                                className={`w-12 h-12 rounded-full flex items-center justify-center 
                                    ${isDisabled ? 'cursor-not-allowed' : 'cursor-pointer'}
                                    ${status === 'completed' ? 'bg-green-500 text-white' : 
                                      status === 'current' ? 'bg-blue-100 text-blue-500 ring-2 ring-blue-500' : 
                                      'bg-gray-100 text-gray-400'}`}
                                whileHover={isDisabled ? {} : { scale: 1.1 }}
                                whileTap={isDisabled ? {} : { scale: 0.95 }}
                                onClick={() => handleStepClick(step.id)}
                            >
                                <Icon className="w-6 h-6" />
                            </motion.div>
                            <motion.div className="flex flex-col items-center mt-2">
                                <motion.span
                                    className={`text-sm font-medium
                                        ${status === 'completed' ? 'text-green-500' : 
                                          status === 'current' ? 'text-blue-700' : 
                                          'text-gray-400'}`}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                >
                                    {step.label}
                                </motion.span>
                                {status === 'completed' && (
                                    <motion.span 
                                        className="text-xs text-green-500"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                    >
                                        Completed
                                    </motion.span>
                                )}
                            </motion.div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default WorkflowHeader;
