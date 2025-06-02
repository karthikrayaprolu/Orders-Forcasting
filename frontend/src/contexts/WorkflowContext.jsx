import React, { createContext, useContext, useState, useEffect } from "react";

// Create the context
const WorkflowContext = createContext();

export const WorkflowProvider = ({ children }) => {
  const [currentStep, setCurrentStep] = useState("home");
  const [database, setDatabase] = useState({
    databaseType: "mongodb",
  });
  const [process, setProcess] = useState({
    timeColumn: "",
    targetVariable: "",
    frequency: "daily",
    features: [],
  });
  const [model, setModel] = useState({
    modelType: "Prophet",
    hyperparameterTuning: false,
    ensembleLearning: false,
    transferLearning: false,
  });
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [availableTables, setAvailableTables] = useState([]);
  const [availableColumns, setAvailableColumns] = useState([]);

  // 🔁 Listen to "change-step" events dispatched from UI
  useEffect(() => {
    const handleStepChange = (e) => {
      if (typeof e.detail === "string") {
        setCurrentStep(e.detail);
      }
    };

    window.addEventListener("change-step", handleStepChange);
    return () => window.removeEventListener("change-step", handleStepChange);
  }, []);

  const value = {
    currentStep,
    setCurrentStep,
    database,
    setDatabase,
    process,
    setProcess,
    model,
    setModel,
    results,
    setResults,
    isLoading,
    setIsLoading,
    availableTables,
    setAvailableTables,
    availableColumns,
    setAvailableColumns,
  };

  return <WorkflowContext.Provider value={value}>{children}</WorkflowContext.Provider>;
};

export const useWorkflow = () => {
  const context = useContext(WorkflowContext);
  if (context === undefined) {
    throw new Error("useWorkflow must be used within a WorkflowProvider");
  }
  return context;
};
