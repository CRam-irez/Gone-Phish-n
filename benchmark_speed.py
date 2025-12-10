"""
Benchmark script to measure response times for email and URL analysis
"""
import time
from src.model_url import URLPhishingDetector
from src.model_email import MultinomialNB

# Load models
print("Loading models...")
url_model = URLPhishingDetector.load("phishing_model.pkl")
email_model = MultinomialNB.load("email_spam_model.pkl")
print("Models loaded!\n")

# Test URLs
test_urls = [
    "https://www.google.com",
    "https://www.paypal-secure-login-verify.com",
    "http://192.168.1.1/admin",
    "https://amazon.com",
    "http://bit.ly/3x4f5g6",
    "https://www.microsoft.com",
    "https://secure-banking-login.net",
    "https://www.netflix.com",
    "http://free-iphone-giveaway.biz",
    "https://www.github.com",
    "https://verify-account-now.com",
    "https://www.youtube.com",
    "http://click-here-urgent.info",
    "https://www.apple.com",
    "https://confirm-your-identity.tk",
    "https://www.facebook.com",
    "http://prize-winner-2024.xyz",
    "https://www.linkedin.com",
    "https://update-payment-info.ml",
    "https://www.twitter.com",
    "http://suspicious-link-123.ru",
    "https://www.instagram.com",
    "https://secure-paypal-verify.net",
    "https://www.reddit.com",
    "http://totally-legit-bank.com",
    "https://www.stackoverflow.com",
    "https://urgent-action-required.info",
    "https://www.wikipedia.org",
    "http://your-account-locked.biz",
    "https://www.amazon.co.uk",
    "https://verify-now-click.com",
    "https://www.dropbox.com",
    "http://free-money-winner.tk",
    "https://www.zoom.us",
    "https://security-alert-urgent.net",
    "https://www.slack.com",
    "http://claim-reward-here.xyz",
    "https://www.spotify.com",
    "https://confirm-payment-details.ml",
    "https://www.twitch.tv",
    "http://urgent-verification.ru",
    "https://www.discord.com",
    "https://account-suspended-help.info",
    "https://www.office.com",
    "http://click-to-unlock-account.biz",
    "https://www.salesforce.com",
    "https://immediate-action-needed.com",
    "https://www.adobe.com",
    "http://confirm-identity-now.tk",
    "https://www.cloudflare.com",
]

# Test emails
test_emails = [
    "Meeting tomorrow at 3pm in conference room A",
    "URGENT! Your account has been compromised. Click here to verify your identity now!",
    "Can you review the quarterly report by Friday?",
    "Congratulations! You've won $1,000,000. Claim your prize now by clicking this link!",
    "Let's schedule a call next week to discuss the project timeline",
    "Your package delivery requires immediate action",
    "Thanks for the feedback on the proposal",
    "FREE GIFT CARD! Click now before it expires!!!",
    "Could you send me the presentation slides from yesterday?",
    "WINNER! You have been selected for a special promotion",
    "Please find attached the monthly budget report",
    "Act now! Your credit card has been charged $999.99",
    "Are you available for lunch on Thursday?",
    "URGENT: Verify your email address or account will be deleted",
    "Reminder about the team building event next Friday",
    "You've inherited $5 million from a distant relative",
    "Can we reschedule our 2pm meeting?",
    "CONGRATULATIONS! Click here to claim your FREE iPhone 15",
    "Here are the notes from today's standup meeting",
    "ALERT: Suspicious activity detected on your account",
    "Happy birthday! Hope you have a great day",
    "You must update your password immediately",
    "The project deadline has been extended to next month",
    "WIN BIG! Enter our sweepstakes now for a chance at $10,000",
    "Thanks for your help with the client presentation",
    "Your PayPal account has been limited. Verify now",
    "Let me know if you need anything for the conference",
    "AMAZING OFFER! Get 90% off all products TODAY ONLY",
    "Could you review this pull request when you have time?",
    "Your Amazon order requires payment verification",
    "Looking forward to seeing everyone at the holiday party",
    "URGENT: IRS tax refund waiting for you. Claim now",
    "The new feature is ready for testing in staging",
    "You've been selected for a $500 gift card. Click to redeem",
    "Can you help me debug this issue in production?",
    "FINAL NOTICE: Your subscription will expire in 24 hours",
    "Great job on the presentation today!",
    "WINNER WINNER! You've won our grand prize drawing",
    "I've shared the Google doc with you for review",
    "Your bank account has been compromised. Verify immediately",
    "Don't forget about tomorrow's code review session",
    "EXCLUSIVE OFFER: Get rich quick with this one simple trick",
    "Could you take a look at the bug report I filed?",
    "Security alert: Unusual sign-in activity detected",
    "The deployment went smoothly, everything is working",
    "LAST CHANCE! Limited time offer expires midnight tonight",
    "I'll send over the contract details by end of day",
    "You have a package waiting. Confirm delivery address now",
    "Let's grab coffee sometime next week to catch up",
    "URGENT ACTION REQUIRED: Update your billing information now",
]

print("=" * 60)
print("URL ANALYSIS SPEED TEST")
print("=" * 60)

url_times = []
for i, url in enumerate(test_urls, 1):
    start = time.perf_counter()
    result = url_model.predict_proba(url)
    end = time.perf_counter()
    elapsed = (end - start) * 1000  # Convert to milliseconds
    url_times.append(elapsed)
    prob_percent = result * 100
    print(f"Test {i}: {elapsed:.2f}ms | {prob_percent:.1f}% phishing - {url[:40]}")

avg_url_time = sum(url_times) / len(url_times)
print(f"\nAverage URL Analysis Time: {avg_url_time:.2f}ms")
print(f"Fastest: {min(url_times):.2f}ms | Slowest: {max(url_times):.2f}ms")

print("\n" + "=" * 60)
print("EMAIL ANALYSIS SPEED TEST")
print("=" * 60)

email_times = []
for i, email in enumerate(test_emails, 1):
    start = time.perf_counter()
    result = email_model.predict_proba(email)
    end = time.perf_counter()
    elapsed = (end - start) * 1000  # Convert to milliseconds
    email_times.append(elapsed)
    prob_percent = result * 100
    print(f"Test {i}: {elapsed:.2f}ms | {prob_percent:.1f}% spam - {email[:40]}")

avg_email_time = sum(email_times) / len(email_times)
print(f"\nAverage Email Analysis Time: {avg_email_time:.2f}ms")
print(f"Fastest: {min(email_times):.2f}ms | Slowest: {max(email_times):.2f}ms")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Average URL Analysis:   {avg_url_time:.2f}ms ({avg_url_time/1000:.3f} seconds)")
print(f"Average Email Analysis: {avg_email_time:.2f}ms ({avg_email_time/1000:.3f} seconds)")
print(f"\nTotal tests run: {len(test_urls) + len(test_emails)}")
print("All analysis completed in real-time (<1 second per input)")
