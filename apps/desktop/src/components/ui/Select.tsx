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
                        "w-full bg-white/5 border border-white/10 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-accent-primary transition-all appearance-none",
                        className
                    )}
                    {...props}
                >
                    {options.map((opt) => (
                        <option key={opt.value} value={opt.value} className="bg-background text-white">
                            {opt.label}
                        </option>
                    ))}
                </select>
                {/* Custom arrow indicator could go here */}
            </div>
        );
    }
);
