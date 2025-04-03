// This script toggles dark mode on and off when the button is clicked.

document.addEventListener("DOMContentLoaded", function () {
    const toggleButton = document.getElementById("dark-mode-toggle");
    const icon = toggleButton.querySelector("i");

    // Load dark mode from localStorage
    const darkModeEnabled = localStorage.getItem("dark-mode") === "enabled";
    if (darkModeEnabled) {
        document.body.classList.add("dark-mode");
        icon.classList.remove("fa-moon");
        icon.classList.add("fa-sun");
    } else {
        icon.classList.remove("fa-sun");
        icon.classList.add("fa-moon");
    }

    // Toggle dark mode on click
    toggleButton.addEventListener("click", function () {
        const isDark = document.body.classList.toggle("dark-mode");
        localStorage.setItem("dark-mode", isDark ? "enabled" : "disabled");

        icon.classList.toggle("fa-moon", !isDark);
        icon.classList.toggle("fa-sun", isDark);
    });
});
