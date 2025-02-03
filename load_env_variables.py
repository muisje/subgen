import os
def load_env_variables(env_filename='subgen.env'):
    """
    Loads environment variables from a specified .env file and sets them.
    """
    try:
        with open(env_filename, 'r') as file:
            for line in file:
                line = line.strip()
                # Skip empty lines or lines starting with a comment
                if not line or line.startswith('#'):
                    continue
                
                # Split line into variable and value, allow comments after the value
                if '#' in line:
                    line = line.split('#', 1)[0].strip()  # Ignore anything after the # symbol
                    
                var, value = line.strip().split('=', 1)
                os.environ[var] = value

        # Set flag to indicate environment variables are loaded
        print(f"Environment variables have been loaded from {env_filename}")

    except FileNotFoundError:
        print(f"{env_filename} file not found. Please run prompt_and_save_env_variables() first.")