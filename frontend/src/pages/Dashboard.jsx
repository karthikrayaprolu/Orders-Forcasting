import React from 'react';
import WorkflowHeader from '../components/WorkflowHeader';
import DataStep from '../steps/DataStep';
import ProcessStep from '../steps/ProcessStep';
import Results from '../steps/Results';

const steps = [
  { id: 0, title: 'Upload Data', description: 'Upload your CSV files' },
  { id: 1, title: 'Configure Forecast', description: 'Set forecast parameters' },
  { id: 2, title: 'Results', description: 'View forecast results' }
];

const Dashboard = ({ currentStep, setCurrentStep, forecastData, onStepComplete }) => {
  const renderStep = () => {
    switch (currentStep) {
      case 0:
        return <DataStep onComplete={() => onStepComplete()} />;
      case 1:
        return <ProcessStep onComplete={(data) => onStepComplete(data)} />;
      case 2:
        return <Results forecasts={forecastData} />;
      default:
        return <DataStep onComplete={() => onStepComplete()} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <WorkflowHeader
          steps={steps}
          currentStep={currentStep}
          onStepClick={(step) => {
            if (step < currentStep) {
              setCurrentStep(step);
            }
          }}
        />
        <main className="py-10">
          {renderStep()}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;