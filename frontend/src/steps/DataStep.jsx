import React, { useState } from 'react';
import { uploadFiles } from '../services/api';
import { motion } from 'framer-motion';
import { FiUpload, FiFile, FiCheckCircle } from 'react-icons/fi';
import { Button } from '../components/ui/button';
import { useWorkflow } from '../contexts/WorkflowContext';
import { toast } from 'sonner';

const DataStep = ({ onComplete }) => {
    const [files, setFiles] = useState({
        header: null,
        items: null,
        workstation: null
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState(false);
    const { completeStep, STEPS } = useWorkflow();

    const handleFileChange = (e, fileType) => {
        const file = e.target.files[0];
        setFiles(prev => ({
            ...prev,
            [fileType]: file
        }));
        setError('');
    };    const handleUpload = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setSuccess(false);

        try {
            // Validate file presence
            if (!files.header || !files.items || !files.workstation) {
                throw new Error('Please select all required files');
            }

            // Validate file types
            const validateFile = (file, expectedName) => {
                if (!file.name.toLowerCase().endsWith('.csv')) {
                    throw new Error(`${expectedName} must be a CSV file`);
                }
            };

            validateFile(files.header, 'Header file');
            validateFile(files.items, 'Items file');
            validateFile(files.workstation, 'Workstation file');

            // Upload files
            const result = await uploadFiles(files.header, files.items, files.workstation);
            
            if (result?.message) {
                setSuccess(true);
                toast.success('Data files uploaded successfully!');
                // Complete this step and move to next
                setTimeout(() => {
                    completeStep(STEPS.DATABASE);
                    toast.success('Moving to process configuration...');
                }, 1500);
            } else {
                throw new Error('Upload response was not in the expected format');
            }
        } catch (err) {
            const errorMessage = err.response?.data?.error || err.message || 'Failed to upload files. Please try again.';
            setError(errorMessage);
            toast.error(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="max-w-4xl mx-auto p-6 bg-white rounded-xl shadow-md border border-gray-100"
        >
            <div className="flex items-center mb-6">
                <div className="p-3 rounded-full bg-blue-100 mr-4">
                    <FiUpload className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                    <h2 className="text-2xl font-bold text-gray-800">Data Upload</h2>
                    <p className="text-gray-500">Upload your CSV Data to begin processing</p>
                </div>
            </div>
            
            <form onSubmit={handleUpload} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {['header', 'items', 'workstation'].map((fileType) => (
                        <div 
                            key={fileType}
                            className={`p-4 border-2 border-dashed rounded-lg transition-all ${
                                files[fileType] 
                                    ? 'border-emerald-200 bg-emerald-50' 
                                    : 'border-gray-200 hover:border-gray-300'
                            }`}
                        >
                            <label className="flex flex-col items-center justify-center cursor-pointer">
                                <div className="mb-3 p-3 rounded-full bg-gray-100">
                                    {files[fileType] ? (
                                        <FiCheckCircle className="h-5 w-5 text-emerald-600" />
                                    ) : (
                                        <FiFile className="h-5 w-5 text-gray-500" />
                                    )}
                                </div>
                                <span className="block text-sm font-medium text-gray-700 capitalize mb-1">
                                    {fileType} Data
                                </span>
                                <span className="block text-xs text-gray-500 mb-3">
                                    {files[fileType]?.name || 'No Data selected'}
                                </span>
                                <div className="relative">
                                    <input
                                        type="file"
                                        accept=".csv"
                                        onChange={(e) => handleFileChange(e, fileType)}
                                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                    />
                                    <Button
                                        variant="outline"
                                        type="button"
                                        className="text-sm"
                                    >
                                        {files[fileType] ? 'Change Data' : 'Select Data'}
                                    </Button>
                                </div>
                            </label>
                        </div>
                    ))}
                </div>

                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="p-3 bg-red-100 text-red-700 rounded-lg text-sm"
                    >
                        {error}
                    </motion.div>
                )}

                {success && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="p-3 bg-emerald-100 text-emerald-700 rounded-lg text-sm"
                    >
                        Data uploaded successfully! Processing...
                    </motion.div>
                )}

                <div className="pt-4">
                    <Button
                        type="submit"
                        disabled={loading || !files.header || !files.items || !files.workstation}
                        className="w-full py-3 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white font-semibold rounded-lg shadow-md transition-all"
                    >
                        {loading ? (
                            <div className="flex items-center justify-center">
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Uploading...
                            </div>
                        ) : (
                            'Upload & Process Data'
                        )}
                    </Button>
                </div>
            </form>
        </motion.div>
    );
};

export default DataStep;