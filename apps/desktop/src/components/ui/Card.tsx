import React from 'react';
import clsx from 'clsx';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    glass?: boolean;
    hoverEffect?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
    ({ className, glass = true, hoverEffect = false, children, ...props }, ref) => {
        return (
            <div
                ref={ref}
                className={clsx(
                    'card',
                    glass && 'glass-card',
                    hoverEffect && 'card-hover',
                    className
                )}
                {...props}
            >
                {children}
            </div>
        );
    }
);
Card.displayName = "Card";
