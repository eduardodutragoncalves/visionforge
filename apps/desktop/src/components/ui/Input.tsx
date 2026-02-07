import React from 'react';
import clsx from 'clsx';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
    ({ className, label, error, ...props }, ref) => {
        return (
            <div className="w-full">
                {label && <label className="block text-sm text-secondary mb-1">{label}</label>}
                <input
                    ref={ref}
                    className={clsx(
                        "w-full bg-graphite-900/80 border border-graphite-700/70 px-3 py-2 text-bone placeholder-haze focus:outline-none focus:ring-2 focus:ring-accent-primary transition-all rounded-xl",
                        error && "border-error focus:ring-error",
                        className
                    )}
                    {...props}
                />
                {error && <span className="text-xs text-error mt-1">{error}</span>}
            </div>
        );
    }
);
