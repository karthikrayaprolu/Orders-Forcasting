import React from "react";

// Simple cn utility if you don't have one
const cn = (...classes) => classes.filter(Boolean).join(" ");

const buttonVariants = ({ 
  variant = "default",
  size = "default",
  className
} = {}) => {
  const baseClasses = "inline-flex items-center justify-center rounded-md font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none";

  const variants = {
    default: "bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600 text-white shadow-lg hover:shadow-xl",
    secondary: "border border-gray-300 bg-white text-gray-700 hover:bg-gray-50",
    outline: "border border-amber-400 text-amber-600 hover:bg-amber-50 hover:border-amber-500",
    accent: "bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-white shadow-lg hover:shadow-xl",
    ghost: "hover:bg-gray-100 text-gray-700",
    link: "text-blue-600 hover:underline underline-offset-4"
  };

  const sizes = {
    default: "px-8 py-6 text-lg",
    sm: "px-6 py-4 text-base",
    lg: "px-10 py-6 text-lg font-bold",
    icon: "p-4"
  };

  return cn(
    baseClasses,
    variants[variant],
    sizes[size],
    className
  );
};

const Button = React.forwardRef(
  (
    {
      className,
      variant = "default",
      size = "default",
      asChild = false,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? "span" : "button";
    return (
      <Comp
        className={buttonVariants({ variant, size, className })}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };