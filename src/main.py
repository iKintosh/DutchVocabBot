#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from data.models import create_tables
from bot import main as run_bot

def setup_environment():
    load_dotenv()
    
    # Check if required environment variables are set
    required_vars = ['TELEGRAM_BOT_TOKEN']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease create a .env file based on .env.example and fill in the required values.")
        return False
    
    return True

def main():
    print("🇳🇱 Dutch Vocabulary Trainer Bot Setup")
    print("=" * 40)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Create database tables
    print("Creating database tables...")
    create_tables()
    print("✅ Database tables created")
    
    # Start the bot
    print("Starting Telegram bot...")
    print("Bot is ready! Send /start to begin learning Dutch! 🚀")
    run_bot()

if __name__ == "__main__":
    main()