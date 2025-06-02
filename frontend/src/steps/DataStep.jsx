import React, { useState } from 'react';
import { uploadFiles } from '../services/api';

const DataStep = ({ onComplete }) => {
    const [files, setFiles] = useState({
        header: null,
        items: null,
        workstation: null
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleFileChange = (e, fileType) => {
        const file = e.target.files[0];
        setFiles(prev => ({
            ...prev,
            [fileType]: file
        }));
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            if (!files.header || !files.items || !files.workstation) {
                throw new Error('Please select all required files');
            }

            await uploadFiles(files.header, files.items, files.workstation);
            onComplete();
        } catch (err) {
            setError(err.message || 'Failed to upload files');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-6">
            <h2 className="text-2xl font-bold mb-6">Upload Data Files</h2>
            
            <form onSubmit={handleUpload} className="space-y-6">
                <div className="space-y-4">
                    {['header', 'items', 'workstation'].map((fileType) => (
                        <div key={fileType}>
                            <label className="block text-sm font-medium text-gray-700 capitalize">
                                {fileType} File
                            </label>
                            <input
                                type="file"
                                accept=".csv"
                                onChange={(e) => handleFileChange(e, fileType)}
                                className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4
                                    file:rounded-md file:border-0 file:text-sm file:font-semibold
                                    file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                            />
                        </div>
                    ))}
                </div>

                {error && (
                    <div className="text-red-600 text-sm">
                        {error}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={loading || !files.header || !files.items || !files.workstation}
                    className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white 
                        ${loading || !files.header || !files.items || !files.workstation
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
                        }`}
                >
                    {loading ? 'Uploading...' : 'Upload Files'}
                </button>
            </form>
        </div>
    );
};

export default DataStep;