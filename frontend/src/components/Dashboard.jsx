import React, { useEffect } from "react";
import { useWorkflow } from "@/contexts/WorkflowContext";
import WorkflowHeader from "./WorkflowHeader";
import DataStep from "../steps/DataStep";
import ProcessStep from "../steps/ProcessStep";
import TrainStep from "../steps/TrainStep";
import ResultsStep from "../steps/Results";
import { gsap } from "gsap";

const Dashboard = () => {
  const { currentStep } = useWorkflow();

  // GSAP timeline for container animations
  useEffect(() => {
    const tl = gsap.timeline();
    
    tl.fromTo(
      ".dashboard-container",
      { opacity: 0, scale: 0.95 },
      { 
        opacity: 1, 
        scale: 1, 
        duration: 0.8, 
        ease: "power3.out",
      }
    );
    
    return () => {
      tl.kill();
    };
  }, []);

  // Render the current step
  const renderStep = () => {
    switch (currentStep) {
      case "database":
        return <DataStep />;
      case "process":
        return <ProcessStep />;
      case "train":
        return <TrainStep />;
      case "results":
        return <ResultsStep />;
      default:
        return <DataStep />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <WorkflowHeader />
      <div className="dashboard-container container mx-auto py-8 px-4">
        <div className="gsap-container">
          {renderStep()}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
