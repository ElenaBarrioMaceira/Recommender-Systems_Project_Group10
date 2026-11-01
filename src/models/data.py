from pydantic import BaseModel, ConfigDict
import pandas as pd
import matplotlib.pyplot as plt

AGE_MAPPING = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

OCCUPATION_MAPPING = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


class UserCharasteristics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    number_of_users: int
    number_of_active_users: int
    age_distribution: pd.DataFrame
    occupation_distribution: pd.DataFrame


class MovieCharasteristics(BaseModel):
    number_of_movies: int
    number_of_movies_reviewed: int


class RatingCharasteristics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    number_of_ratings: int
    mean_rating: float
    median_rating: float
    rating_distribution: pd.DataFrame


class DataCharasteristics(BaseModel):
    density: float
    sparsity: float


class TimeCharasteristics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    first_interaction: pd.Timestamp
    last_interaction: pd.Timestamp
    time_span: pd.Timedelta
    interactions_per_month: pd.Series


class Interactions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    interactions_per_user: pd.Series
    interactions_per_item: pd.Series
    interactions_per_time: pd.Series


class Data(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_data: pd.DataFrame
    movie_data: pd.DataFrame
    rating_data: pd.DataFrame


class DataForModeling(Data):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class DataWithCharasteristics(Data):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data_charasteristics: DataCharasteristics
    time_charasteristics: TimeCharasteristics
    user_charasteristics: UserCharasteristics
    rating_charasteristics: RatingCharasteristics
    movie_charasteristics: MovieCharasteristics
    interactions: Interactions

    def print_summary(self):
        user = self.user_charasteristics
        movie = self.movie_charasteristics
        rating = self.rating_charasteristics
        data = self.data_charasteristics
        first_interaction = self.time_charasteristics.first_interaction
        last_interaction = self.time_charasteristics.last_interaction
        time_span = self.time_charasteristics.time_span
        interactions_per_month = self.time_charasteristics.interactions_per_month
        interactions_per_user = self.interactions.interactions_per_user
        interactions_per_item = self.interactions.interactions_per_item

        print("\n")
        print("--------------------User charasteristics--------------------")
        print(f"Number of users: {user.number_of_users}")
        print(f"Number of active users: {user.number_of_active_users}")
        print("\nUsers age distribution:")
        print(user.age_distribution.rename(index=AGE_MAPPING))
        print("\nUsers occupation distribution:")
        print(user.occupation_distribution.rename(index=OCCUPATION_MAPPING))

        print("\n\n--------------------Movie charasteristics--------------------")
        print(f"\nNumber of movies: {movie.number_of_movies}")
        print(f"Number of movies reviewed: {movie.number_of_movies_reviewed}")

        print("\n\n--------------------Rating charasteristics--------------------")
        print(f"\nNumber of ratings: {rating.number_of_ratings}")
        print(f"Mean rating: {rating.mean_rating:.2f}")
        print(f"Median rating: {rating.median_rating:.2f}")

        print("\n\n--------------------Time charasteristics--------------------")
        print(f"\nFirst interaction: {first_interaction.strftime('%d/%m/%Y')}")
        print(f"Last interaction: {last_interaction.strftime('%d/%m/%Y')}")
        print(f"Time span in days: {time_span.days}")
        print(f"Interactions per month: {interactions_per_month}")

        print("\n\n--------------------Data charasteristics--------------------")
        print(f"\nDensity of data: {data.density:.2f}")
        print(f"Sparsity of data: {data.sparsity:.2f}")

        print("\n\n---------- Interaction Characteristics ----------")
        print(f"Total interactions: {len(self.rating_data)}")
        print("\nInteractions per user:")
        print(interactions_per_user.describe().round(2))
        print("\nInteractions per item:")
        print(interactions_per_item.describe().round(2))

    def plot_interaction_distributions(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Interactions per user
        axes[0].hist(
            self.interactions.interactions_per_user,
            bins=50,
            edgecolor="black",
            alpha=0.7,
        )
        axes[0].set_title("Distribution of Interactions per User")
        axes[0].set_xlabel("Number of Interactions")
        axes[0].set_ylabel("Number of Users")
        axes[0].grid(alpha=0.3)

        # Interactions per item
        axes[1].hist(
            self.interactions.interactions_per_item,
            bins=50,
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_title("Distribution of Interactions per Item")
        axes[1].set_xlabel("Number of Interactions")
        axes[1].set_ylabel("Number of Items")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_rating_distribution(self):
        """Plot the distribution of rating values."""
        distribution = self.rating_charasteristics.rating_distribution.sort_index()

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(
            distribution.index.astype(str),
            distribution["Count"],
            edgecolor="black",
            alpha=0.7,
        )

        ax.bar_label(bars, padding=3, fmt="%.0f")

        ax.set_title("Distribution of Ratings")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Ratings")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        fig.tight_layout()
        plt.show()

    def plot_time_distribution(self):
        """Plot monthly activity over the complete observed interaction time span."""
        time = self.time_charasteristics
        monthly = self.interactions.interactions_per_time.sort_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(
            monthly.index,
            monthly.values,
            width=25,
            align="center",
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_title(
            "Interactions Over Time\n"
            f"{time.first_interaction:%Y-%m-%d} to {time.last_interaction:%Y-%m-%d} "
            f"({time.time_span.days:,} days)"
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of interactions")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()
        return fig, ax
