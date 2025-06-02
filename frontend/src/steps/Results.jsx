import React from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const Results = ({ forecasts }) => {
    if (!forecasts || forecasts.length === 0) {
        return (
            <div className="max-w-4xl mx-auto p-6">
                <p className="text-gray-500">No forecast data available.</p>
            </div>
        );
    }

    const dates = forecasts.map(f => new Date(f.date).toLocaleDateString());
    const targets = Object.keys(forecasts[0]).filter(key => key !== 'date');

    const colors = {
        orders: 'rgb(255, 99, 132)',
        products: 'rgb(54, 162, 235)',
        employees: 'rgb(75, 192, 192)',
        throughput: 'rgb(153, 102, 255)'
    };

    const charts = targets.map(target => {
        const data = {
            labels: dates,
            datasets: [
                {
                    label: target.charAt(0).toUpperCase() + target.slice(1),
                    data: forecasts.map(f => f[target]),
                    borderColor: colors[target],
                    backgroundColor: colors[target],
                    tension: 0.1
                }
            ]
        };

        const options = {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: `${target.charAt(0).toUpperCase() + target.slice(1)} Forecast`
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        };

        return (
            <div key={target} className="mb-8">
                <Line options={options} data={data} />
            </div>
        );
    });

    return (
        <div className="max-w-4xl mx-auto p-6">
            <h2 className="text-2xl font-bold mb-6">Forecast Results</h2>
            <div className="space-y-8">
                {charts}
            </div>
        </div>
    );
};

export default Results;