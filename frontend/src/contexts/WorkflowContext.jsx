import React, { createContext, useContext, useState, useEffect } from "react";

// Create the context
const WorkflowContext = createContext({});

export const WorkflowProvider = ({ children }) => {
  const STEPS = {
    DATABASE: "database",
    PROCESS: "process",
    TRAIN: "train",
    RESULTS: "results",
  };

  // Core workflow states
  const [currentStep, setCurrentStep] = useState(STEPS.DATABASE);
  const [completedSteps, setCompletedSteps] = useState(new Set());
  
  // Data and model states
  const [uploadedFiles, setUploadedFiles] = useState(null);
  const [process, setProcess] = useState({
    horizon: 30,
    timePeriod: 'day',
    aggregationMethod: 'mean',
    selectedTargets: []
  });
  
  // Model configuration states
  const [modelSelections, setModelSelections] = useState({});
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  // Save state to localStorage when it changes
  useEffect(() => {
    if (process) {
      localStorage.setItem('forecastProcess', JSON.stringify(process));
    }
    if (modelSelections) {
      localStorage.setItem('modelSelections', JSON.stringify(modelSelections));
    }
    if (results) {
      localStorage.setItem('forecastResults', JSON.stringify(results));
    }
  }, [process, modelSelections, results]);
  // Load saved state on initial mount
  useEffect(() => {
    const savedProcess = localStorage.getItem('forecastProcess');
    const savedSelections = localStorage.getItem('modelSelections');
    const savedResults = localStorage.getItem('forecastResults');

    if (savedProcess) {
      setProcess(JSON.parse(savedProcess));
    }
    if (savedSelections) {
      setModelSelections(JSON.parse(savedSelections));
    }
    if (savedResults) {
      setResults(JSON.parse(savedResults));
    }
  }, []);

  const completeStep = (step) => {
    setCompletedSteps((prev) => new Set([...prev, step]));
    // Move to next step
    switch (step) {
      case STEPS.DATABASE:
        setCurrentStep(STEPS.PROCESS);
        break;
      case STEPS.PROCESS:
        setCurrentStep(STEPS.TRAIN);
        break;
      case STEPS.TRAIN:
        setCurrentStep(STEPS.RESULTS);
        break;
      default:
        break;
    }
  };

  const resetWorkflow = () => {
    setCurrentStep(STEPS.DATABASE);
    setCompletedSteps(new Set());
    setUploadedFiles(null);
    setProcess({
      horizon: 30,
      timePeriod: 'day',
      aggregationMethod: 'mean',
      selectedTargets: []
    });
    setModelSelections({});
    setResults(null);
    setError(null);
    localStorage.removeItem('forecastProcess');
    localStorage.removeItem('modelSelections');
  };

  const canAccessStep = (step) => {
    const stepOrder = Object.values(STEPS);
    const currentStepIndex = stepOrder.indexOf(currentStep);
    const targetStepIndex = stepOrder.indexOf(step);
    return targetStepIndex <= currentStepIndex || completedSteps.has(step);
  };

  const value = {
    // Step management
    currentStep,
    setCurrentStep,
    completedSteps,
    completeStep,
    canAccessStep,
    resetWorkflow,
    STEPS,

    // Data and process management
    uploadedFiles,
    setUploadedFiles,
    process,
    setProcess,

    // Model and results management
    modelSelections,
    setModelSelections,
    results,
    setResults,

    // UI states
    isLoading,
    setIsLoading,
    error,
    setError
  };

  return (
    <WorkflowContext.Provider value={value}>
      {children}
    </WorkflowContext.Provider>
  );
};

export const useWorkflow = () => {
  const context = useContext(WorkflowContext);
  if (context === undefined) {
    throw new Error("useWorkflow must be used within a WorkflowProvider");
  }
  return context;
};