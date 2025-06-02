import React from 'react';

const WorkflowHeader = ({ steps, currentStep, onStepClick }) => {
  return (
    <nav className="py-4">
      <ol className="flex items-center w-full">
        {steps.map((step, index) => (
          <li 
            key={step.id} 
            className={`flex items-center ${
              index < steps.length - 1 ? 'w-full' : ''
            }`}
          >
            <button
              onClick={() => onStepClick(step.id)}
              className={`flex items-center ${
                step.id <= currentStep
                  ? 'text-blue-600 hover:text-blue-900'
                  : 'text-gray-400 cursor-not-allowed'
              }`}
              disabled={step.id > currentStep}
            >
              <span className={`
                flex items-center justify-center w-8 h-8 border-2 rounded-full
                ${step.id < currentStep ? 'border-blue-600 bg-blue-600 text-white' : ''}
                ${step.id === currentStep ? 'border-blue-600 text-blue-600' : ''}
                ${step.id > currentStep ? 'border-gray-300 text-gray-400' : ''}
              `}>
                {step.id + 1}
              </span>
              <span className="ml-2 text-sm font-medium">
                {step.title}
              </span>
            </button>

            {index < steps.length - 1 && (
              <div className="w-full flex items-center">
                <div className={`h-0.5 w-full mx-4 ${
                  step.id < currentStep ? 'bg-blue-600' : 'bg-gray-200'
                }`}></div>
              </div>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
};

export default WorkflowHeader;