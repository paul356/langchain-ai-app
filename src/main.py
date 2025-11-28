"""
Main entry point for the LangChain AI Application.
Provides a menu to access different features.
"""

import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

load_dotenv()


def show_menu():
    """Display the main menu."""
    print("\n" + "=" * 50)
    print("🤖 LangChain AI Application")
    print("=" * 50)
    print("1. Simple Chat Bot")
    print("2. Advanced Prompting Demo")
    print("3. Vector Memory Chat (Session Management)")
    print("4. Exit")
    print("-" * 50)


def main():
    """Main application loop."""
    while True:
        show_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            print("\n🚀 Starting Simple Chat Bot...")
            try:
                from chat.simple_chat import main as chat_main
                chat_main()
            except Exception as e:
                print(f"❌ Error starting chat bot: {e}")

        elif choice == "2":
            print("\n🧠 Running Advanced Prompting Demo...")
            try:
                from chat.advanced_prompts import demo_advanced_features
                demo_advanced_features()
            except Exception as e:
                print(f"❌ Error running demo: {e}")

        elif choice == "3":
            print("\n💾 Starting Vector Memory Chat...")
            try:
                from chat.memory_chat import main as memory_main
                memory_main()
            except Exception as e:
                print(f"❌ Error starting memory chat: {e}")

        elif choice == "4":
            print("\n👋 Thank you for using LangChain AI Application!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()