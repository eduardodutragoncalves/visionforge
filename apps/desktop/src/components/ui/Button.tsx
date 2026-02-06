import React, { ButtonHTMLAttributes } from 'react';
import clsx from 'clsx';


interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'glass';
    size?: 'sm' | 'md' | 'lg';
    isLoading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'primary', size = 'md', isLoading, children, ...props }, ref) => {

        // Using vanilla CSS modules or utility classes would be ideal if setup, 
        // but since we updated index.css with vars, we'll use inline styles or standard classes 
        // combined with the variables. 
        // However, to keep it clean and performant, I'll use standard classes defined here that map to those vars.
        // For this environment, I'll rely on a style object approach or simple class mapping if Tailwind isn't fully configured.
        // Given the prompt mentioned "tokens defined" in index.css, I will construct classes that use those tokens.
        // IMPORTANT: Since I didn't setup Tailwind in this step (it was optional/vanilla preferred in prompt),
        // I will write a small CSS module or just scoped styles? 
        // Actually, I can just use the `style` prop for dynamic things or create a `components.css` file.
        // But to be self-contained, I will use "CSS-in-JS" style objects using the vars or just className string constructions
        // assuming we can add a few atomic classes to index.css or just use `style`.

        // Better approach: I'll add component classes to index.css or a new components.css
        // For now, let's assume I'll append to index.css for these components or use `style`.
        // Let's use `style` with our vars for simplicity in this file, or better, 
        // since I can't easily edit index.css constantly, I'll use a style object.

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
