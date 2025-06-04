import React from 'react';
import WorkflowHeader from '../components/WorkflowHeader';
import DataStep from '../steps/DataStep';
import ProcessStep from '../steps/ProcessStep';
import TrainStep from '../steps/TrainStep';
import Results from '../steps/Results';
import { useWorkflow } from '../contexts/WorkflowContext';

const steps = [
  { id: 0, title: 'Upload Data', description: 'Upload your CSV files' },
  { id: 1, title: 'Configure Forecast', description: 'Set forecast parameters' },
  { id: 2, title: 'Train Models', description: 'Configure and train models' },
  { id: 3, title: 'Results', description: 'View forecast results' }
];

const Dashboard = () => {
  const { currentStep, setCurrentStep, canAccessStep, STEPS } = useWorkflow();

  const renderStep = () => {
    switch (currentStep) {
      case STEPS.DATABASE:
        return <DataStep />;
      case STEPS.PROCESS:
        return <ProcessStep />;
      case STEPS.TRAIN:
        return <TrainStep />;
      case STEPS.RESULTS:
        return <Results />;
      default:
        return <DataStep />;
    }
  };

  const handleStepClick = (stepId) => {
    if (canAccessStep(stepId)) {
      setCurrentStep(stepId);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <WorkflowHeader
          steps={steps}
          currentStep={currentStep}
          onStepClick={handleStepClick}
        />
        <main className="py-10">
          {renderStep()}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;