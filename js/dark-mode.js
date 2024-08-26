// dark-mode.js

// Get the current local time
const currentHour = new Date().getHours();

// Check if the time is between 7 PM (19:00) and 7 AM (07:00)
if (currentHour >= 19 || currentHour < 7) {
    document.body.classList.add('dark-mode');
}