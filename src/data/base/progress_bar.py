from tqdm import tqdm # type: ignore

class ProgressBar:
    def __init__(self, total_steps: int = 0):
        self.bar = tqdm(total=total_steps)

    def check(self, steps: int = 1) -> None:
        """
        Updates the progress of the bar by `steps`.
        
        Args:
            steps (int): steps for the progress bar to move forward.
        Returns:
            None.
        """
        self.bar.update(steps)

    def update_total_steps(self, total_steps: int) -> None:
        """
        Updates the total steps of the progress bar.
        
        Args:
            total_steps (int): the new total steps for the progress bar.
        Returns:
            None.
        """
        self.bar.total = total_steps
        self.bar.refresh()
        
    def log(self, message: str) -> None:
        """
        Sets the message in the console.
        
        Args:
            message (str): string that will be shown in console, next to the
            progress bar.
        Returns:
            None.
        """
        self.bar.set_description(message)
        
    def close(self) -> None:
        """
        Closes the progres bar.
        
        Args:
            None.
        Returns:
            None.
        """
        self.bar.close()
