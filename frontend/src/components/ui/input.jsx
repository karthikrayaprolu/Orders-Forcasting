import React from 'react';
import { cn } from '@/lib/utils';

const Input = React.forwardRef(
  (
    {
      className,
      type = 'text',
      variant = 'default',
      size = 'default',
      label,
      helperText,
      error,
      icon,
      iconPosition = 'left',
      fullWidth = true,
      ...props
    },
    ref
  ) => {
    const inputId = React.useId();

    const variants = {
      default: 'border-input bg-background',
      outline: 'border border-gray-300 bg-transparent',
      filled: 'border-b border-gray-300 bg-gray-50',
      ghost: 'border-0 bg-transparent',
      error: 'border-red-500 bg-red-50',
    };

    const sizes = {
      sm: 'h-8 px-2 text-sm',
      default: 'h-10 px-3 text-base',
      lg: 'h-12 px-4 text-lg',
    };

    return (
      <div className={cn('grid gap-1', fullWidth && 'w-full')}>
        {label && (
          <label
            htmlFor={inputId}
            className={cn(
              'text-sm font-medium',
              error ? 'text-red-600' : 'text-gray-700'
            )}
          >
            {label}
          </label>
        )}

        <div className="relative flex items-center">
          {icon && iconPosition === 'left' && (
            <span className="absolute left-3 text-gray-400">
              {icon}
            </span>
          )}

          <input
            id={inputId}
            type={type}
            className={cn(
              'flex w-full rounded-md py-2 transition-colors',
              'file:border-0 file:bg-transparent file:text-sm file:font-medium',
              'placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
              'disabled:cursor-not-allowed disabled:opacity-50',
              icon && iconPosition === 'left' ? 'pl-10' : 'pl-3',
              icon && iconPosition === 'right' ? 'pr-10' : 'pr-3',
              variants[variant] || variants.default,
              sizes[size] || sizes.default,
              error && variants.error,
              className
            )}
            ref={ref}
            {...props}
          />

          {icon && iconPosition === 'right' && (
            <span className="absolute right-3 text-gray-400">
              {icon}
            </span>
          )}
        </div>

        {helperText && (
          <p
            className={cn(
              'text-sm',
              error ? 'text-red-600' : 'text-gray-500'
            )}
          >
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export { Input };