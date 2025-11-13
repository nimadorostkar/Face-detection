"""
Database Utility Scripts

Helper scripts for managing the face recognition database.
"""

import sys
from database import FaceDatabase


def show_stats():
    """Show database statistics."""
    try:
        db = FaceDatabase()
        
        # Get person count
        with db.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM persons")
            person_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM face_encodings")
            encoding_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM recognition_logs")
            log_count = cur.fetchone()[0]
        
        print("\n" + "="*50)
        print("Database Statistics")
        print("="*50)
        print(f"Persons: {person_count}")
        print(f"Face Encodings: {encoding_count}")
        print(f"Recognition Logs: {log_count}")
        print("="*50)
        
        # Show recent recognitions
        if log_count > 0:
            print("\nRecent Recognitions (last 10):")
            logs = db.get_recognition_stats(limit=10)
            for log in logs:
                name = log['recognized_name'] or 'Unknown'
                confidence = log['confidence'] or 0.0
                print(f"  - {name} (confidence: {confidence:.2f}) at {log['recognized_at']}")
        
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def list_persons():
    """List all persons in the database."""
    try:
        db = FaceDatabase()
        
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT p.id, p.name, COUNT(fe.id) as encoding_count, p.created_at
                FROM persons p
                LEFT JOIN face_encodings fe ON p.id = fe.person_id
                GROUP BY p.id, p.name, p.created_at
                ORDER BY p.name
            """)
            
            persons = cur.fetchall()
            
            if not persons:
                print("No persons in database.")
            else:
                print("\n" + "="*50)
                print("Persons in Database")
                print("="*50)
                for person_id, name, encoding_count, created_at in persons:
                    print(f"ID: {person_id} | Name: {name} | Encodings: {encoding_count} | Created: {created_at}")
                print("="*50)
        
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def delete_person(person_id: int):
    """Delete a person and all their data."""
    try:
        db = FaceDatabase()
        db.delete_person(person_id)
        db.close()
        print(f"✓ Person {person_id} deleted successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python db_utils.py stats          # Show database statistics")
        print("  python db_utils.py list            # List all persons")
        print("  python db_utils.py delete <id>     # Delete a person by ID")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "stats":
        show_stats()
    elif command == "list":
        list_persons()
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Error: Please provide person ID")
            sys.exit(1)
        try:
            person_id = int(sys.argv[2])
            delete_person(person_id)
        except ValueError:
            print("Error: Person ID must be a number")
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

