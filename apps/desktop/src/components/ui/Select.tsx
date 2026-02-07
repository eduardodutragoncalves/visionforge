import React from 'react';
import clsx from 'clsx';

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
    label?: string;
    options: { value: string; label: string }[];
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
    ({ className, label, options, ...props }, ref) => {
        return (
            <div className="w-full">
                {label && <label className="block text-sm text-secondary mb-1">{label}</label>}
                <select
                    ref={ref}
                    className={clsx(
                        "w-full bg-graphite-900/80 border border-graphite-700/70 px-3 py-2 text-bone focus:outline-none focus:ring-2 focus:ring-accent-primary transition-all appearance-none rounded-xl",
                        className
                    )}
                    {...props}
                >
                    {options.map((opt) => (
                        <option key={opt.value} value={opt.value} className="bg-graphite-900 text-bone">
                            {opt.label}
                        </option>
                    ))}
                </select>
                {/* Custom arrow indicator could go here */}
            </div>
        );
    }
);
