// 1. Feature engineering - process of using domain knowledge to extract features from raw data via data mining techniques.
// 2. Data Collection & Data Verificaton - at least we want to know what data did we get and what our model marked as spam.
// 3. Model creation - combine data and ideas.
// 4. Serving infrastructure - how to perform inference?

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/furdarius/talk-go-tensorflow/runners"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type User struct {
	Name      string
	Email     string
	CreatedAt time.Time
}

type Message struct {
	User      User
	CreatedAt time.Time
	Text      string
}

func main() {
	oneMinAgo := time.Now().Add(-1 * time.Minute)
	twoMinAgo := time.Now().Add(-2 * time.Minute)

	msg := Message{
		User:      User{Name: "John", Email: "10minutemail.com", CreatedAt: twoMinAgo},
		CreatedAt: oneMinAgo,
		Text:      "You have WON a guaranteed 1000 cash! To get your money, click the link http://russianroulette.com?n=QJKGIGHJJGCBL",
	}

	dir := "./model"
	model, err := loadSavedModel(dir)
	if err != nil {
		panic(err)
	}

	runner := runners.NewSignatureRunner(model.Graph, model.Session)

	err = runner.LoadOperations()
	if err != nil {
		panic(err)
	}

	req := runners.SignatureRequest{
		MsgCreatedAt:  msg.CreatedAt.Unix(),
		MsgText:       msg.Text,
		UserCreatedAt: msg.User.CreatedAt.Unix(),
		UserEmail:     msg.User.Email,
	}

	resp, err := runner.Run(context.Background(), req)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Spam confidence: %f", resp.Confidence)
}

func loadSavedModel(dir string) (*tf.SavedModel, error) {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return nil, fmt.Errorf("directory \"%s\" doesn't exist", dir)
	}

	metaGraph := "serve"

	model, err := tf.LoadSavedModel(dir, []string{metaGraph}, &tf.SessionOptions{})
	if err != nil {
		return nil, err
	}

	return model, nil
}
