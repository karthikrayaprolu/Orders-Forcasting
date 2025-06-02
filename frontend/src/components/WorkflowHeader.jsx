import React from 'react';
import { Home as HomeIcon, ChevronRight } from 'lucide-react';

const WorkflowHeader = ({ steps, currentStep, onStepClick, onHomeClick }) => {
  return (
    <header className="bg-[#0f172a] text-white px-6 py-4 shadow flex items-center justify-between">
      <button 
        onClick={onHomeClick}
        className="flex items-center gap-2 text-sm font-medium hover:text-yellow-300 transition"
      >
        <HomeIcon className="w-5 h-5" />
        Home
      </button>

      <div className="flex-1 flex items-center justify-center">
        {steps.map((step, index) => (
          <React.Fragment key={step.id}>
            <div
              onClick={() => onStepClick(step.id)}
              className={`flex flex-col items-center px-2 cursor-pointer ${
                step.id <= currentStep ? 'text-yellow-300' : 'text-gray-400'
              }`}
            >
              <div
                className={`w-8 h-8 rounded-full text-sm flex items-center justify-center font-bold ${
                  step.id <= currentStep ? 'bg-yellow-300 text-black' : 'bg-gray-500 text-white'
                }`}
              >
                {step.id + 1}
              </div>
              <span className="mt-1 text-xs">{step.title}</span>
            </div>

            {index < steps.length - 1 && (
              <ChevronRight className="mx-2 text-gray-500" />
            )}
          </React.Fragment>
        ))}
      </div>

      <div className="w-20"></div> {/* spacer */}
    </header>
  );
};

export default WorkflowHeader;
