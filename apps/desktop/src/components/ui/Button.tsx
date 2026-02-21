import React, { ButtonHTMLAttributes } from 'react';
import clsx from 'clsx';


interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'glass';
    size?: 'sm' | 'md' | 'lg';
    isLoading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'primary', size = 'md', isLoading, children, ...props }, ref) => {
        return (
            <button
                ref={ref}
                className={clsx('btn', `btn-${variant}`, `btn-${size}`, className)}
                disabled={isLoading || props.disabled}
                {...props}
            >
                {isLoading ? <span className="loader" /> : children}
            </button>
        );
    }
);
Button.displayName = "Button";
